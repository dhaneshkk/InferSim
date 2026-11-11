from config.model_config import ModelConfig
from hardware.gpu import GPU, transfer_time
from utils.parallel import ep_size, gpus_per_node


class Comm:
    """Per-layer communication model.

    See utils/parallel.py for the parallelism layout. Two kinds of traffic:
      * EP traffic (dispatch/combine or the all_reduce stand-in) around a MoE
        FFN, across the ``ep_size = world_size // tp_size`` TP groups. Dense
        FFNs and ep_size == 1 need none (attention is data parallel).
      * TP all_reduce inside a TP group, twice per layer (after attention and
        after the FFN) when tp_size > 1.
    """

    def __init__(
        self,
        config: ModelConfig,
        gpu: GPU,
        world_size: int,
        num_nodes=1,
        enable_deepep=False,
        tp_size=1,
    ):
        self.config = config
        self.gpu = gpu
        self.world_size = world_size
        self.num_nodes = num_nodes
        self.enable_deepep = enable_deepep
        self.tp_size = tp_size
        self.ep_size = ep_size(world_size, tp_size)
        self.gpus_per_node = gpus_per_node(world_size, num_nodes)

    def size_bw_model(self, tensor_shape, use_fp8=False, inter_node=False):
        if self.ep_size <= 1:
            return 0
        size = 1 if use_fp8 else 2
        for v in tensor_shape:
            size *= v
        bw = self.gpu.rdma_bw if inter_node else self.gpu.nvlink_bw
        return transfer_time(size, bw)

    def all_reduce(self, num_tokens):
        tensor_shape = [num_tokens * self.ep_size, self.config.hidden_size]
        return self.size_bw_model(
            tensor_shape, use_fp8=False, inter_node=(self.num_nodes > 1)
        )

    def dispatch(self, num_tokens, mode="normal"):
        if mode == "normal":
            send_tokens = num_tokens * (self.num_nodes - 1)
            tensor_shape1 = [send_tokens, self.config.hidden_size]
            t1 = self.size_bw_model(tensor_shape1, use_fp8=True, inter_node=True)

            tensor_shape2 = [num_tokens, self.config.hidden_size]
            t2 = self.size_bw_model(tensor_shape2, use_fp8=True, inter_node=False)
            return t1 + t2
        else:
            send_tokens = num_tokens * self.config.num_experts_per_tok
            tensor_shape = [send_tokens, self.config.hidden_size]
            return self.size_bw_model(
                tensor_shape, use_fp8=True, inter_node=(self.num_nodes > 1)
            )

    def combine(self, num_tokens, mode="normal"):
        if mode == "normal":
            rcv_tokens = num_tokens * (self.num_nodes - 1)
            tensor_shape1 = [rcv_tokens, self.config.hidden_size]
            t1 = self.size_bw_model(tensor_shape1, use_fp8=False, inter_node=True)

            tensor_shape2 = [num_tokens, self.config.hidden_size]
            t2 = self.size_bw_model(tensor_shape2, use_fp8=False, inter_node=False)
            return t1 + t2
        else:
            rcv_tokens = num_tokens * self.config.num_experts_per_tok
            tensor_shape = [rcv_tokens, self.config.hidden_size]
            return self.size_bw_model(
                tensor_shape, use_fp8=False, inter_node=(self.num_nodes > 1)
            )

    def a2f(self, num_tokens):
        tensor_shape = [num_tokens, self.config.hidden_size]
        return self.size_bw_model(tensor_shape, use_fp8=True, inter_node=True)

    def f2a(self, num_tokens):
        tensor_shape = [num_tokens, self.config.hidden_size]
        return self.size_bw_model(tensor_shape, use_fp8=False, inter_node=True)

    def _ep_comm(self, num_tokens, deepep_mode):
        # Dense FFN layers are replicated across DP groups: no EP traffic.
        if not self.config.is_moe or self.ep_size <= 1:
            return 0, 0
        if self.enable_deepep:
            return self.dispatch(num_tokens, deepep_mode), self.combine(
                num_tokens, deepep_mode
            )
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)

    def prefill_comm(self, num_tokens: int):
        """EP traffic (before FFN, after FFN) for one layer, in seconds."""
        return self._ep_comm(num_tokens, "normal")

    def decode_comm(self, num_tokens: int):
        """EP traffic (before FFN, after FFN) for one layer, in seconds."""
        return self._ep_comm(num_tokens, "low_latency")

    def tp_all_reduce(self, num_tokens: int, tp_size: int = None):
        """Time of ONE ring all_reduce of the layer output inside a TP group.

        Callers charge it twice per layer (after attention and after FFN).
        Ring all_reduce moves 2 * (N-1)/N of the tensor per GPU. The TP group
        uses NVLink when it fits in one node and RDMA when it spans nodes.
        """
        if tp_size is None:
            tp_size = self.tp_size
        if tp_size <= 1:
            return 0
        size = 2 * num_tokens * self.config.hidden_size  # bf16 output
        transfer_size = size * 2 * (tp_size - 1) / tp_size
        bw = self.gpu.nvlink_bw if tp_size <= self.gpus_per_node else self.gpu.rdma_bw
        return transfer_time(transfer_size, bw)
