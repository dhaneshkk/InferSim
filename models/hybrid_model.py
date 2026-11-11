import math

from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import BYTES_PER_GIB, gpu_map
from kvcache.kvcache import get_kvcache_size, get_states_size
from layers.attn import create_attention
from layers.linear_attn import create_linear_attn
from models.common import (
    HYBRID_DECODE_SCHEDULER_OVERHEAD_MS,
    PREFILL_SCHEDULER_OVERHEAD_MS,
    ffn_group_times,
    ffn_params_gib_per_gpu,
    reserved_kvcache_gib,
    tp_all_reduce_time,
)
from params.params import (
    get_attn_params_size,
    get_expert_params_size,
    get_linear_attn_params_size,
)
from utils.diagnostics import warn


class HybridModel:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        tp_size = self.args.tp_size
        full_attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        linear_attn_params_bytes = get_linear_attn_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm, tp_size
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One full attn params size (MB):", full_attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One linear attn params size (MB):",
                linear_attn_params_bytes / 1024 / 1024,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        # Split rule: see utils/parallel.py
        params_per_gpu = ffn_params_gib_per_gpu(
            self.config, self.args.use_fp8_gemm, self.args.world_size, tp_size
        )
        params_per_gpu += (
            self.config.num_full_attn_layers * full_attn_params_bytes
            + self.config.num_linear_attn_layers * linear_attn_params_bytes
        ) / BYTES_PER_GIB
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
        target_kvcache_bytes = self.kvcache_mem * BYTES_PER_GIB / target_bs
        kvcache_bytes = (
            get_kvcache_size(self.config, self.args.use_fp8_kv, self.args.tp_size)
            / self.config.num_hidden_layers
        )
        kvcache_bytes *= self.config.num_full_attn_layers * context_len
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-req KV cache size (MB):", target_kvcache_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req full attn KV cache size (MB):",
                kvcache_bytes / 1024 / 1024,
            )
        )
        states_bytes = get_states_size(self.config, self.args.tp_size)
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req states size (MB):", states_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req cache size (MB):",
                (kvcache_bytes + states_bytes) / 1024 / 1024,
            )
        )
        if kvcache_bytes + states_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
            warn("KV cache + states for the target batch do not fit in GPU memory")
        self.kvcache_bytes = kvcache_bytes / context_len
        self.states_bytes = states_bytes
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
        moe_gflops = get_moe_gflops(self.config, self.args.tp_size)
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer full attn core (GFLOPs):", attn_core_gflops
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
                "Per-token full attn core (GFLOPs):",
                attn_core_gflops * self.config.num_full_attn_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token others (GFLOPs):",
                other_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (attn_core_gflops + moe_gflops + other_gflops)
                * self.config.num_hidden_layers,
            )
        )

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens)
        )
        # full attn
        full_attn = create_attention(
            self.config,
            self.args.use_fp8_gemm,
            self.args.use_fp8_kv,
            self.args.tp_size,
            getattr(self.args, "prefill_attn_mfu", None),
        )
        t_full_attn_core = full_attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        t_full_attn_others = full_attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )
        # A chunk carries chunk/isl of a full sequence's attention cost
        # (same rule as Model.prefill).
        t_full_attn_core *= self.args.max_prefill_tokens / self.args.target_isl

        # linear attn
        linear_attn = create_linear_attn(
            self.config, self.args.use_fp8_gemm, self.args.tp_size
        )
        t_linear_attn_core = linear_attn.prefill_attn_core(
            self.args.target_isl, self.states_bytes, self.args.device_type
        )
        t_linear_attn_core *= self.args.max_prefill_tokens / self.args.target_isl
        t_linear_attn_others = linear_attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )

        ffn_groups = ffn_group_times(
            self.args, self.gpu, self.config, "prefill", self.args.max_prefill_tokens
        )
        tp_comm = tp_all_reduce_time(
            self.args, self.gpu, self.config, self.args.max_prefill_tokens
        )

        num_tokens = self.args.max_prefill_tokens
        ttft = (
            t_full_attn_core + t_full_attn_others
        ) * self.config.num_full_attn_layers
        ttft += (
            t_linear_attn_core + t_linear_attn_others
        ) * self.config.num_linear_attn_layers
        for n, ffn_t, c1, c2 in ffn_groups:
            ttft += n * (ffn_t + c1 + c2)
        ttft += 2 * tp_comm * self.config.num_hidden_layers
        ttft *= 1000  # convert to ms
        ttft += PREFILL_SCHEDULER_OVERHEAD_MS

        tgs = num_tokens / self.args.tp_size / (ttft / 1000)
        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
        print("{:<40} {:<10.0f}".format("Throughput (TGS:tok/GPU/s):", tgs))
        return {"ttft_ms": ttft, "prefill_tgs": tgs}

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        # full attn
        full_attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv, self.args.tp_size
        )
        t_full_attn_core = full_attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        t_full_attn_others = full_attn.decode_attn_others(
            self.target_bs, self.args.device_type
        )

        # linear attn
        linear_attn = create_linear_attn(
            self.config, self.args.use_fp8_gemm, self.args.tp_size
        )
        t_linear_attn_core = linear_attn.decode_attn_core(
            self.target_bs, self.states_bytes, self.args.device_type
        )
        t_linear_attn_others = linear_attn.decode_attn_others(
            self.target_bs, self.args.device_type
        )

        ffn_groups = ffn_group_times(
            self.args,
            self.gpu,
            self.config,
            "decode",
            self.target_bs,
            overlap_shared_expert=getattr(
                self.args, "enable_shared_expert_overlap", False
            ),
        )
        tp_comm = tp_all_reduce_time(self.args, self.gpu, self.config, self.target_bs)

        num_tokens = self.target_bs
        tpot = (
            t_full_attn_core + t_full_attn_others
        ) * self.config.num_full_attn_layers
        tpot += (
            t_linear_attn_core + t_linear_attn_others
        ) * self.config.num_linear_attn_layers
        for n, ffn_t, c1, c2 in ffn_groups:
            tpot += n * (ffn_t + c1 + c2)
        tpot += 2 * tp_comm * self.config.num_hidden_layers
        tpot *= 1000  # convert to ms
        scheduler_overhead_ms = getattr(
            self.args, "decode_scheduler_overhead_ms", None
        )
        if scheduler_overhead_ms is None:
            scheduler_overhead_ms = HYBRID_DECODE_SCHEDULER_OVERHEAD_MS
        tpot += scheduler_overhead_ms

        tgs = num_tokens / self.args.tp_size / (tpot / 1000)
        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        print("{:<40} {:<10.0f}".format("Throughput (TGS:tok/GPU/s):", tgs))
        if tpot > self.args.target_tpot:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
            warn(f"TPOT {tpot:.2f} ms exceeds target {self.args.target_tpot} ms")
        return {"tpot_ms": tpot, "decode_tgs": tgs}
