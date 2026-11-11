import copy
import json


def _first_present(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


class ModelConfig:
    def __init__(
        self,
        config_path,
    ):
        d = dict()
        with open(config_path, "r") as f:
            d = json.load(f)

        root_model_type = d["model_type"]
        if root_model_type in ["qwen3_5", "qwen3_5_moe"]:
            d = d["text_config"]

        self.model_type = d["model_type"]
        self.is_qwen3_5_moe = (
            root_model_type == "qwen3_5_moe"
            and self.model_type == "qwen3_5_moe_text"
        )

        self.hidden_size = d["hidden_size"]
        self.num_hidden_layers = d["num_hidden_layers"]

        # Non-hybrid models use full attention in every hidden layer.
        self.num_full_attn_layers = self.num_hidden_layers
        self.num_linear_attn_layers = 0
        self.is_hybrid_linear = d.get("full_attention_interval") is not None
        if self.is_hybrid_linear:
            layer_types = d.get("layer_types")
            if layer_types:
                self.num_full_attn_layers = sum(
                    1 for t in layer_types if t == "full_attention"
                )
            else:
                self.num_full_attn_layers = (
                    self.num_hidden_layers // d["full_attention_interval"]
                )
            self.num_linear_attn_layers = (
                self.num_hidden_layers - self.num_full_attn_layers
            )
            self.linear_conv_kernel_dim = d["linear_conv_kernel_dim"]
            self.linear_key_head_dim = d["linear_key_head_dim"]
            self.linear_num_key_heads = d["linear_num_key_heads"]
            self.linear_value_head_dim = d["linear_value_head_dim"]
            self.linear_num_value_heads = d["linear_num_value_heads"]

        self.attn_type = "MHA/GQA"
        if "kv_lora_rank" in d:
            self.attn_type = "MLA"

        # attn
        self.attn_output_gate = d.get("attn_output_gate", False)
        self.index_topk = None
        if self.attn_type == "MHA/GQA":
            self.num_attention_heads = d["num_attention_heads"]
            self.num_key_value_heads = d["num_key_value_heads"]
            if "head_dim" in d:
                self.head_dim = d["head_dim"]
            else:
                self.head_dim = self.hidden_size // self.num_attention_heads
        elif self.attn_type == "MLA":
            self.q_lora_rank = d["q_lora_rank"]
            self.qk_nope_head_dim = d["qk_nope_head_dim"]
            self.qk_rope_head_dim = d["qk_rope_head_dim"]
            self.kv_lora_rank = d["kv_lora_rank"]
            self.num_attention_heads = d["num_attention_heads"]
            self.v_head_dim = d["v_head_dim"]
            self.index_topk = d.get("index_topk")
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # FFN/MoE
        # HF configs name the expert counts differently per family:
        #   DeepSeek: n_routed_experts / n_shared_experts
        #   Qwen:     num_experts / shared_expert_intermediate_size
        num_routed = _first_present(
            d, ["num_routed_experts", "n_routed_experts", "num_experts"]
        )
        self.is_moe = num_routed is not None and num_routed > 1
        if num_routed is None and "moe_intermediate_size" in d:
            raise ValueError(
                f"{config_path}: has moe_intermediate_size but no expert count "
                "(expected one of num_routed_experts, n_routed_experts, "
                "num_experts)"
            )

        # Dense FFN width, used by dense models and by the dense layers of
        # MoE models (e.g. DeepSeek first_k_dense_replace).
        self.dense_intermediate_size = d.get("intermediate_size")

        if self.is_moe:
            self.num_routed_experts = num_routed
            self.num_experts_per_tok = d["num_experts_per_tok"]
            self.intermediate_size = d["moe_intermediate_size"]
            self.shared_expert_intermediate_size = d.get(
                "shared_expert_intermediate_size"
            )
            if self.shared_expert_intermediate_size is not None:
                if self.shared_expert_intermediate_size % self.intermediate_size != 0:
                    raise ValueError(
                        "InferSim currently requires shared_expert_intermediate_size "
                        "to be a multiple of moe_intermediate_size"
                    )
                self.num_shared_experts = (
                    self.shared_expert_intermediate_size // self.intermediate_size
                )
            else:
                self.num_shared_experts = _first_present(
                    d, ["num_shared_experts", "n_shared_experts"], 0
                )
                self.shared_expert_intermediate_size = (
                    self.num_shared_experts * self.intermediate_size
                )
            self.num_moe_layers = self._count_moe_layers(d)
        else:
            self.num_routed_experts = 1
            self.num_experts_per_tok = 1
            self.intermediate_size = d["intermediate_size"]
            self.num_shared_experts = 0
            self.shared_expert_intermediate_size = 0
            self.num_moe_layers = 0

        # Layers whose FFN is a dense MLP rather than MoE.
        self.num_dense_ffn_layers = self.num_hidden_layers - self.num_moe_layers
        if self.is_moe and self.num_dense_ffn_layers > 0:
            if not self.dense_intermediate_size:
                raise ValueError(
                    f"{config_path}: {self.num_dense_ffn_layers} dense FFN layers "
                    "but no intermediate_size"
                )

    def _count_moe_layers(self, d):
        """Count layers that use the MoE FFN.

        Follows the HF modeling code of each family:
          DeepSeek: MoE iff layer >= first_k_dense_replace and
                    layer % moe_layer_freq == 0
          Qwen MoE: MoE iff layer not in mlp_only_layers and
                    (layer + 1) % decoder_sparse_step == 0
        """
        first_k_dense = d.get("first_k_dense_replace", 0) or 0
        moe_layer_freq = d.get("moe_layer_freq", 1) or 1
        mlp_only = set(d.get("mlp_only_layers", []) or [])
        sparse_step = d.get("decoder_sparse_step", 1) or 1
        count = 0
        for i in range(self.num_hidden_layers):
            if i < first_k_dense or i % moe_layer_freq != 0:
                continue
            if i in mlp_only or (i + 1) % sparse_step != 0:
                continue
            count += 1
        return count

    def dense_ffn_view(self):
        """A copy of this config describing the dense FFN layers.

        MoE models with dense layers (DeepSeek's first_k_dense_replace) cost
        those layers as a dense MLP of ``dense_intermediate_size``.
        """
        view = copy.copy(self)
        view.is_moe = False
        view.num_routed_experts = 1
        view.num_experts_per_tok = 1
        view.num_shared_experts = 0
        view.shared_expert_intermediate_size = 0
        view.intermediate_size = self.dense_intermediate_size
        return view
