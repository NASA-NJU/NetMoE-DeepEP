import argparse
import random
from typing import Tuple

import torch
import torch.distributed as dist

import deep_ep
import deep_gemm
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import calc_diff
from deep_gemm.utils import per_block_cast_to_fp8


def create_grouped_fp8_weight(num_groups: int, n: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn((num_groups, n, k), dtype=torch.bfloat16, device='cuda') / (k**0.5)
    fp8_data, scales = [], []
    for group_idx in range(num_groups):
        data, scale = per_block_cast_to_fp8(weight[group_idx], use_ue8m0=False)
        fp8_data.append(data)
        scales.append(scale)
    return torch.stack(fp8_data), torch.stack(scales)


def test_sbo(rank: int,
             num_ranks: int,
             group: dist.ProcessGroup,
             buffer: deep_ep.Buffer,
             num_tokens: int,
             hidden: int,
             num_experts: int,
             num_topk: int,
             num_sbo_sms: int) -> None:
    torch.manual_seed(1234 + rank)
    random.seed(1234 + rank)

    num_local_experts = num_experts // num_ranks
    capacity = num_tokens * num_ranks
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=True).indices.to(deep_ep.topk_idx_t)
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    packed_recv_x, packed_recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=True,
        async_finish=False,
        return_recv_hook=False)
    packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous())

    weights = create_grouped_fp8_weight(num_local_experts, hidden, hidden)
    expected_m = max(1, min(capacity, (num_tokens * num_ranks * num_topk + num_experts - 1) // num_experts))

    baseline_gemm_out = torch.empty((num_local_experts, capacity, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        packed_recv_x,
        weights,
        baseline_gemm_out,
        packed_recv_count,
        expected_m,
        disable_ue8m0_cast=True)
    baseline_combined, _, _ = buffer.low_latency_combine(
        baseline_gemm_out,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
        return_recv_hook=False)

    # Allocate for the smallest supported block so the actual JIT-selected layout always fits.
    max_signal_stride = (capacity + 15) // 16
    comp_signal = torch.zeros(num_local_experts * max_signal_stride, dtype=torch.int32, device='cuda')
    sbo_gemm_out = torch.empty_like(baseline_gemm_out)
    gemm_stream = torch.cuda.Stream()
    gemm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(gemm_stream):
        sbo_config = deep_gemm.m_grouped_fp8_gemm_nt_sbo_masked(
            packed_recv_x,
            weights,
            sbo_gemm_out,
            packed_recv_count,
            expected_m,
            disable_ue8m0_cast=True,
            send_signal=comp_signal.data_ptr())
    assert sbo_config is not None, 'DeepGEMM SBO requires an SM90 kernel'
    block_m, threshold = sbo_config

    sbo_combined, _, recv_hook = buffer.low_latency_combine(
        sbo_gemm_out,
        topk_idx,
        topk_weights,
        handle,
        zero_copy=False,
        return_recv_hook=True,
        overlap=True,
        packed_recv_count=packed_recv_count,
        comp_signal=comp_signal,
        block_m=block_m,
        threshold=threshold,
        num_sms=num_sbo_sms)
    recv_hook()
    torch.cuda.synchronize()

    diff = calc_diff(baseline_combined, sbo_combined)
    assert torch.isnan(sbo_combined).sum().item() == 0
    assert diff < 1e-5, f'SBO result differs from baseline: {diff=}'
    if rank == 0:
        print(f'SBO correctness passed: {block_m=}, {threshold=}, {diff=:.3e}', flush=True)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert args.num_experts % num_ranks == 0

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(args.num_tokens, args.hidden, num_ranks, args.num_experts)
    buffer = deep_ep.Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=args.num_experts // num_ranks,
        allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
        explicitly_destroy=True)
    test_sbo(rank,
             num_ranks,
             group,
             buffer,
             args.num_tokens,
             args.hidden,
             args.num_experts,
             args.num_topk,
             args.num_sbo_sms)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DeepGEMM and DeepEP single-QP SBO overlap')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-sbo-sms', type=int, default=4)
    parser.add_argument('--disable-nvlink', action='store_true')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
