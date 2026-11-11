import math

from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import BYTES_PER_GIB, gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from models.common import (
    DECODE_SCHEDULER_OVERHEAD_MS,
    PREFILL_SCHEDULER_OVERHEAD_MS,
    ffn_group_times,
    ffn_layer_groups,
    ffn_params_gib_per_gpu,
    reserved_kvcache_gib,
    tp_all_reduce_time,
)
from params.params import get_attn_params_size, get_expert_params_size
from utils.diagnostics import warn


class Model:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        tp_size = self.args.tp_size
        attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        # Split rule (see utils/parallel.py): attention and dense FFN are TP
        # sharded and DP replicated; routed experts are EP sharded over
        # world_size // tp_size groups.
        params_per_gpu = (
            attn_params_bytes * self.config.num_hidden_layers / BYTES_PER_GIB
        )
        params_per_gpu += ffn_params_gib_per_gpu(
            self.config, self.args.use_fp8_gemm, self.args.world_size, tp_size
        )
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))
        self.kvcache_mem = reserved_kvcache_gib(self.gpu.mem, params_per_gpu)

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))
        context_len = self.args.target_isl + self.args.target_osl

        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))
        target_kvcache_bytes = (
            self.kvcache_mem * BYTES_PER_GIB / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(
            self.config, self.args.use_fp8_kv, self.args.tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache_bytes / 1024
            )
        )
        if kvcache_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
            warn("KV cache for the target batch does not fit in GPU memory")
        self.kvcache_bytes = kvcache_bytes
        self.target_bs = target_bs

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers)
        )
        # per-token per-layer gflops
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, self.args.tp_size, absorb=True
        )
        groups = ffn_layer_groups(self.config)
        ffn_total_gflops = sum(
            n * get_moe_gflops(c, self.args.tp_size) for _, c, n in groups
        )
        moe_gflops = ffn_total_gflops / self.config.num_hidden_layers
        if self.config.is_moe and self.config.num_dense_ffn_layers > 0:
            print(
                "{:<40} {:<10}".format(
                    "MoE / dense FFN layers:",
                    f"{self.config.num_moe_layers} / {self.config.num_dense_ffn_layers}",
                )
            )
        if getattr(self.config, "index_topk", None):
            warn(
                "sparse attention (index_topk) is not modelled; "
                "attention is costed as dense MLA"
            )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer others (GFLOPs):", other_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token attn core (GFLOPs):",
                attn_core_gflops * self.config.num_hidden_layers,
            )
        )
        print("{:<40} {:<10.2f}".format("Per-token MoE (GFLOPs):", ffn_total_gflops))
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token others (GFLOPs):",
                other_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (attn_core_gflops + other_gflops) * self.config.num_hidden_layers
                + ffn_total_gflops,
            )
        )

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        num_layers = self.config.num_hidden_layers
        chunk = self.args.max_prefill_tokens
        print("{:<40} {:<10}".format("Max prefill tokens:", chunk))
        attn = create_attention(
            self.config,
            self.args.use_fp8_gemm,
            self.args.use_fp8_kv,
            self.args.tp_size,
            getattr(self.args, "prefill_attn_mfu", None),
        )
        attn_core_time = attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        attn_other_time = attn.prefill_attn_others(chunk, self.args.device_type)
        # A chunk of `chunk` tokens carries chunk/isl of a full sequence's
        # attention cost, whether it packs several short sequences
        # (isl < chunk) or is one slice of a long one (isl > chunk).
        attn_core_time *= chunk / self.args.target_isl
        attn_time = attn_core_time + attn_other_time

        ffn_groups = ffn_group_times(
            self.args, self.gpu, self.config, "prefill", chunk
        )
        tp_comm = tp_all_reduce_time(self.args, self.gpu, self.config, chunk)

        num_tokens = chunk
        ttft = 0.0
        if self.args.enable_tbo:
            num_tokens *= 2
            sm = self.args.sm_ratio
            for n, ffn_t, c1, c2 in ffn_groups:
                layer = max(attn_time / sm, c1) + max(attn_time / sm, c2)
                layer += max(ffn_t / sm, c1) + max(ffn_t / sm, c2)
                ttft += n * layer
            # TP all_reduces are on the critical path of both micro-batches.
            ttft += num_layers * 2 * 2 * tp_comm
        else:
            ttft += num_layers * attn_time
            for n, ffn_t, c1, c2 in ffn_groups:
                ttft += n * (ffn_t + c1 + c2)
            ttft += num_layers * 2 * tp_comm  # after attention and after FFN
        ttft *= 1000  # convert to ms
        ttft += PREFILL_SCHEDULER_OVERHEAD_MS

        tgs = num_tokens / self.args.tp_size / (ttft / 1000)
        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
        print("{:<40} {:<10.0f}".format("Throughput (TGS:tok/GPU/s):", tgs))
        return {"ttft_ms": ttft, "prefill_tgs": tgs}

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        num_layers = self.config.num_hidden_layers
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv, self.args.tp_size
        )
        attn_core_time = attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        attn_other_time = attn.decode_attn_others(self.target_bs, self.args.device_type)
        attn_time = attn_core_time + attn_other_time

        ffn_groups = ffn_group_times(
            self.args, self.gpu, self.config, "decode", self.target_bs
        )
        tp_comm = tp_all_reduce_time(self.args, self.gpu, self.config, self.target_bs)

        num_tokens = self.target_bs
        tpot = 0.0
        if self.args.enable_tbo:
            num_tokens *= 2
            for n, ffn_t, c1, c2 in ffn_groups:
                tpot += n * 2 * max(attn_time, ffn_t + c1 + c2)
            tpot += num_layers * 2 * 2 * tp_comm
        else:
            tpot += num_layers * attn_time
            for n, ffn_t, c1, c2 in ffn_groups:
                tpot += n * (ffn_t + c1 + c2)
            tpot += num_layers * 2 * tp_comm  # after attention and after FFN
        tpot *= 1000  # convert to ms
        scheduler_overhead_ms = getattr(
            self.args, "decode_scheduler_overhead_ms", None
        )
        if scheduler_overhead_ms is None:
            scheduler_overhead_ms = DECODE_SCHEDULER_OVERHEAD_MS
        tpot += scheduler_overhead_ms

        tgs = num_tokens / self.args.tp_size / (tpot / 1000)
        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        print("{:<40} {:<10.0f}".format("Throughput (TGS:tok/GPU/s):", tgs))
        if tpot > self.args.target_tpot:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
            warn(f"TPOT {tpot:.2f} ms exceeds target {self.args.target_tpot} ms")
        return {"tpot_ms": tpot, "decode_tgs": tgs}
