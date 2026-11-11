from config.model_config import ModelConfig
from hardware.gpu import GPU, transfer_time
from utils.parallel import ep_size, tp_intermediate, tp_num_heads, tp_num_kv_heads


def get_mha_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards heads; hidden_size is NOT sharded (row/col-parallel + allreduce)
    heads = tp_num_heads(config.num_attention_heads, tp_size)
    kv_heads = tp_num_kv_heads(config.num_key_value_heads, tp_size)
    q_multiplier = 2 if config.attn_output_gate else 1
    wq = config.hidden_size * q_multiplier * heads * config.head_dim
    wk = config.hidden_size * kv_heads * config.head_dim
    wv = config.hidden_size * kv_heads * config.head_dim
    wo = config.hidden_size * heads * config.head_dim
    if use_fp8:
        return wq + wk + wv + wo
    return 2 * (wq + wk + wv + wo)


def get_mla_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards attention heads; hidden_size and lora ranks are NOT sharded
    heads = tp_num_heads(config.num_attention_heads, tp_size)
    wq_down = config.hidden_size * config.q_lora_rank
    wq_up = config.q_lora_rank * heads * config.qk_head_dim
    wkv_down = config.hidden_size * config.kv_lora_rank
    wkv_up = config.kv_lora_rank * heads * (config.qk_nope_head_dim + config.v_head_dim)
    wo = config.hidden_size * heads * config.v_head_dim
    if use_fp8:
        return wq_down + wq_up + wkv_down + wkv_up + wo
    return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)


def get_gdn_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards linear-attn key/value heads; hidden_size is NOT sharded
    k_heads = tp_num_heads(config.linear_num_key_heads, tp_size)
    v_heads = tp_num_heads(config.linear_num_value_heads, tp_size)
    key_dim = k_heads * config.linear_key_head_dim
    value_dim = v_heads * config.linear_value_head_dim
    # in_proj_qkvz + in_proj_ba
    wq = config.hidden_size * key_dim
    wk = wq
    wv = config.hidden_size * value_dim
    wz = wv
    wa = config.hidden_size * v_heads
    wb = wa
    # out_proj: value_dim -> hidden
    wo = value_dim * config.hidden_size
    gemm_params = wq + wk + wv + wz + wa + wb + wo
    # depthwise conv over q, k, v channels; kept in bf16 even for FP8 GEMMs
    wconv = (2 * key_dim + value_dim) * config.linear_conv_kernel_dim
    gemm_bytes = gemm_params if use_fp8 else 2 * gemm_params
    return gemm_bytes + 2 * wconv


def get_attn_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    if config.attn_type == "MHA/GQA":
        return get_mha_params_size(config, use_fp8, tp_size)
    elif config.attn_type == "MLA":
        return get_mla_params_size(config, use_fp8, tp_size)


def get_linear_attn_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    return get_gdn_params_size(config, use_fp8, tp_size)


def get_expert_params_size(config: ModelConfig, use_fp8: bool, tp_size: int):
    # TP shards intermediate_size; hidden_size is NOT sharded
    w = 3 * config.hidden_size * tp_intermediate(config.intermediate_size, tp_size)
    if not use_fp8:
        w *= 2
    return w


def get_ffn_params_per_layer(
    config: ModelConfig, use_fp8: bool, world_size: int, tp_size: int
):
    """Bytes of FFN weights one GPU holds for one MoE (or dense) layer."""
    expert_bytes = get_expert_params_size(config, use_fp8, tp_size)
    if not config.is_moe:
        # Dense FFN is TP-sharded and replicated across DP groups.
        return expert_bytes
    local_routed = config.num_routed_experts / ep_size(world_size, tp_size)
    return expert_bytes * (config.num_shared_experts + local_routed)


def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, tp_size: int):
    size = get_attn_params_size(config, use_fp8, tp_size)
    return transfer_time(size, gpu.mem_bw)


def get_expected_active_experts(
    config: ModelConfig, num_gpus: int, tp_size: int, num_tokens=None
):
    """Estimate local routed experts touched by a uniformly routed batch."""
    if not config.is_moe:
        return 1
    num_local_experts = config.num_routed_experts / ep_size(num_gpus, tp_size)
    if num_tokens is None:
        return num_local_experts

    topk = min(config.num_experts_per_tok, num_local_experts)
    probability_not_selected = (1 - topk / num_local_experts) ** num_tokens
    return num_local_experts * (1 - probability_not_selected)


def load_moe_weights_time(
    config: ModelConfig,
    use_fp8: bool,
    gpu: GPU,
    num_gpus,
    tp_size=1,
    num_tokens=None,
):
    size = get_expert_params_size(config, use_fp8, tp_size)
    size *= get_expected_active_experts(config, num_gpus, tp_size, num_tokens)
    return transfer_time(size, gpu.mem_bw)
