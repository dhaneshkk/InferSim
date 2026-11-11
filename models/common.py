"""Pieces shared by Model and HybridModel.

Keeping these in one place stops the two model classes from drifting apart
(they previously disagreed on communication, prefill chunking and constants).
"""

from comm.comm import Comm
from hardware.gpu import BYTES_PER_GIB
from layers.moe import MoE
from params.params import get_ffn_params_per_layer

# GPU memory held back from the KV cache, in GiB.
RUNTIME_RESERVED_GIB = 15  # CUDA context, activations, NCCL buffers, graphs
ENCODER_RESERVED_GIB = 5  # vision/audio encoder of multimodal checkpoints

# Fixed per-step CPU/scheduler costs calibrated against SGLang.
PREFILL_SCHEDULER_OVERHEAD_MS = 30
DECODE_SCHEDULER_OVERHEAD_MS = 5  # Model
HYBRID_DECODE_SCHEDULER_OVERHEAD_MS = 2  # HybridModel


def ffn_layer_groups(config):
    """[(label, layer_config, num_layers)] for every distinct FFN layer type.

    MoE models can contain dense FFN layers (DeepSeek first_k_dense_replace,
    Qwen mlp_only_layers); those are costed as a dense MLP.
    """
    if not config.is_moe:
        return [("Dense FFN", config, config.num_hidden_layers)]
    groups = [("MoE", config, config.num_moe_layers)]
    if config.num_dense_ffn_layers > 0:
        groups.append(
            ("Dense FFN", config.dense_ffn_view(), config.num_dense_ffn_layers)
        )
    return [g for g in groups if g[2] > 0]


def ffn_params_gib_per_gpu(config, use_fp8, world_size, tp_size):
    total = 0
    for _, layer_config, num_layers in ffn_layer_groups(config):
        total += num_layers * get_ffn_params_per_layer(
            layer_config, use_fp8, world_size, tp_size
        )
    return total / BYTES_PER_GIB


def reserved_kvcache_gib(gpu_mem_gib, params_gib):
    kvcache_mem = gpu_mem_gib - params_gib - RUNTIME_RESERVED_GIB - ENCODER_RESERVED_GIB
    if kvcache_mem <= 0:
        raise ValueError(
            f"Weights ({params_gib:.2f} GiB) plus reserved memory "
            f"({RUNTIME_RESERVED_GIB + ENCODER_RESERVED_GIB} GiB) exceed GPU memory "
            f"({gpu_mem_gib} GiB); increase world_size/tp_size or use FP8."
        )
    return kvcache_mem


def ffn_group_times(args, gpu, config, phase, num_tokens, overlap_shared_expert=False):
    """Per-layer FFN and EP-communication times for each FFN layer type.

    Returns [(num_layers, ffn_time, comm_before, comm_after)] in seconds.
    """
    results = []
    groups = ffn_layer_groups(config)
    for label, layer_config, num_layers in groups:
        if len(groups) > 1:
            print("{:<40} {:<10}".format(f"[{label} layers]:", num_layers))
        if phase == "prefill":
            moe = MoE(
                layer_config,
                args.use_fp8_gemm,
                args.tp_size,
                getattr(args, "prefill_moe_mfu", None),
            )
            t = moe.prefill_moe(num_tokens, args.device_type, args.world_size)
        else:
            moe = MoE(layer_config, args.use_fp8_gemm, args.tp_size)
            t = moe.decode_moe(
                num_tokens,
                args.device_type,
                args.world_size,
                overlap_shared_expert=overlap_shared_expert,
            )
        comm = Comm(
            layer_config,
            gpu,
            args.world_size,
            args.num_nodes,
            args.enable_deepep,
            args.tp_size,
        )
        if phase == "prefill":
            c1, c2 = comm.prefill_comm(num_tokens)
        else:
            c1, c2 = comm.decode_comm(num_tokens)
        if layer_config.is_moe:
            print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", c1 * 1e6))
            print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", c2 * 1e6))
        results.append((num_layers, t, c1, c2))
    return results


def tp_all_reduce_time(args, gpu, config, num_tokens):
    """One TP all_reduce of the layer output (charge it twice per layer)."""
    comm = Comm(
        config, gpu, args.world_size, args.num_nodes, args.enable_deepep, args.tp_size
    )
    t = comm.tp_all_reduce(num_tokens)
    if args.tp_size > 1:
        print("{:<40} {:<10.2f}".format("TP all_reduce (us):", t * 1e6))
    return t
