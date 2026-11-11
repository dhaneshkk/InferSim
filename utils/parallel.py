"""Single source of truth for how a layer is sharded.

Parallelism model used everywhere in InferSim:
  * Attention: tensor parallel over ``tp_size``; data parallel across the
    ``world_size // tp_size`` TP groups.
  * Dense FFN: tensor parallel over ``tp_size``; replicated across TP groups
    (pure data parallel), so it needs no cross-group communication.
  * MoE FFN: intermediate size tensor parallel over ``tp_size``; routed
    experts expert-parallel over ``ep_size = world_size // tp_size`` groups.
    Shared experts are replicated on every EP rank.
"""


def tp_num_heads(num_heads, tp_size):
    if num_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads={num_heads} is not divisible by tp_size={tp_size}"
        )
    return num_heads // tp_size


def tp_num_kv_heads(num_kv_heads, tp_size):
    """KV heads held by one TP rank.

    When tp_size > num_kv_heads the KV heads are replicated (as in vLLM and
    SGLang): each rank still holds one full KV head, so the per-GPU KV cache
    never drops to zero.
    """
    if tp_size <= num_kv_heads:
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={num_kv_heads} is not divisible by "
                f"tp_size={tp_size}"
            )
        return num_kv_heads // tp_size
    if tp_size % num_kv_heads != 0:
        raise ValueError(
            f"tp_size={tp_size} must be a multiple of "
            f"num_key_value_heads={num_kv_heads} to replicate KV heads"
        )
    return 1


def ep_size(world_size, tp_size):
    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size={world_size} is not divisible by tp_size={tp_size}"
        )
    return world_size // tp_size


def tp_intermediate(intermediate_size, tp_size):
    if intermediate_size % tp_size != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} is not divisible by "
            f"tp_size={tp_size}"
        )
    return intermediate_size // tp_size


def gpus_per_node(world_size, num_nodes):
    if world_size % num_nodes != 0:
        raise ValueError(
            f"world_size={world_size} is not divisible by num_nodes={num_nodes}"
        )
    return world_size // num_nodes
