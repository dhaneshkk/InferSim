"""Regression tests for the issues found in the code review.

Each test is named after the bug it pins down. Run from the repo root:
    python3 -m unittest discover -s tests
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from comm.comm import Comm
from config.model_config import ModelConfig
from hardware.gpu import BYTES_PER_GIB, gpu_map, transfer_time
from kvcache.kvcache import get_kvcache_size
from layers.attn import MLA
from mfu.mfu import combine_up_down_mfu, get_gemm_mfu
from models.common import ffn_layer_groups, ffn_params_gib_per_gpu
from params.params import get_gdn_params_size
from utils import diagnostics
from utils.parallel import tp_num_kv_heads

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(name):
    return ModelConfig(os.path.join(ROOT, "hf_configs", name))


def run_main(*args):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "main.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def args(**kw):
    base = dict(
        world_size=1,
        tp_size=1,
        num_nodes=1,
        enable_deepep=False,
        use_fp8_gemm=False,
        device_type="H20",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class ConfigParsingTest(unittest.TestCase):
    def test_every_bundled_config_parses_expected_ffn_type(self):
        expected_moe = {
            "deepseek_v3_config.json": (True, 256, 1),
            "deepseek_v3.2_config.json": (True, 256, 1),
            "qwen3-30B-A3B_config.json": (True, 128, 0),
            "qwen3-235B-A22B_config.json": (True, 128, 0),
            "qwen3-next-80B-A3B_config.json": (True, 512, 1),
            "qwen3.5-35B-A3B_config.json": (True, 256, 1),
            "qwen3.5-122B-A10B_config.json": (True, 256, 1),
            "qwen3-8B_config.json": (False, 1, 0),
        }
        for name, (is_moe, routed, shared) in expected_moe.items():
            with self.subTest(name):
                c = cfg(name)
                self.assertEqual(c.is_moe, is_moe)
                self.assertEqual(c.num_routed_experts, routed)
                self.assertEqual(c.num_shared_experts, shared)

    def test_deepseek_n_prefixed_keys_are_moe(self):
        # Bug 1: real DeepSeek configs use n_routed_experts / n_shared_experts.
        c = cfg("deepseek_v3.2_config.json")
        self.assertTrue(c.is_moe)
        self.assertEqual(c.intermediate_size, 2048)

    def test_first_k_dense_replace_layers_are_dense(self):
        # Bug 2: DeepSeek's first 3 layers are dense 18432-wide MLPs.
        c = cfg("deepseek_v3_config.json")
        self.assertEqual((c.num_moe_layers, c.num_dense_ffn_layers), (58, 3))
        groups = {label: (lc, n) for label, lc, n in ffn_layer_groups(c)}
        self.assertEqual(groups["MoE"][1], 58)
        self.assertEqual(groups["Dense FFN"][1], 3)
        self.assertEqual(groups["Dense FFN"][0].intermediate_size, 18432)
        self.assertFalse(groups["Dense FFN"][0].is_moe)

    def test_moe_config_without_expert_count_fails_loudly(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            with open(
                os.path.join(ROOT, "hf_configs/qwen3-30B-A3B_config.json")
            ) as src:
                d = json.load(src)
            d.pop("num_experts")
            json.dump(d, f)
        try:
            with self.assertRaises(ValueError):
                ModelConfig(f.name)
        finally:
            os.unlink(f.name)


class ParallelismTest(unittest.TestCase):
    def test_dense_weights_do_not_shrink_with_data_parallel_world_size(self):
        # Bug 3: dense FFN was split by world_size as if it were EP.
        c = cfg("qwen3-8B_config.json")
        sizes = {ffn_params_gib_per_gpu(c, False, w, 1) for w in (1, 2, 8)}
        self.assertEqual(len(sizes), 1)

    def test_moe_weights_still_split_by_ep(self):
        c = cfg("qwen3-30B-A3B_config.json")
        one = ffn_params_gib_per_gpu(c, False, 1, 1)
        four = ffn_params_gib_per_gpu(c, False, 4, 1)
        self.assertAlmostEqual(one / four, 4.0)

    def test_dense_dp_has_no_ep_comm(self):
        # Bug 4: pure DP dense model was charged two all_reduces per layer.
        c = cfg("qwen3-8B_config.json")
        comm = Comm(c, gpu_map["H20"], world_size=2)
        self.assertEqual(comm.decode_comm(64), (0, 0))

    def test_pure_tp_has_no_ep_comm(self):
        # Bug 4: pure TP was charged EP all_reduce on top of TP all_reduce.
        c = cfg("qwen3-30B-A3B_config.json")
        comm = Comm(c, gpu_map["H20"], world_size=4, tp_size=4)
        self.assertEqual(comm.decode_comm(64), (0, 0))
        self.assertGreater(comm.tp_all_reduce(64), 0)

    def test_tp_all_reduce_charged_twice_per_layer(self):
        c = cfg("qwen3-8B_config.json")
        common = [
            "--config-path",
            "hf_configs/qwen3-8B_config.json",
            "--decode-bs",
            "64",
            "--decode-only",
            "--world-size",
            "2",
        ]
        tp1 = run_main(*common)
        tp2 = run_main(*common, "--tp-size", "2")
        self.assertEqual(tp1.returncode, 0, tp1.stderr)
        self.assertEqual(tp2.returncode, 0, tp2.stderr)
        self.assertNotIn("TP all_reduce", tp1.stdout)
        comm = Comm(c, gpu_map["H20"], world_size=2, tp_size=2)
        self.assertGreater(comm.tp_all_reduce(64), 0)

    def test_tp_spanning_nodes_uses_rdma(self):
        c = cfg("qwen3-8B_config.json")
        gpu = gpu_map["H20"]
        intra = Comm(c, gpu, world_size=8, num_nodes=1, tp_size=8).tp_all_reduce(64)
        inter = Comm(c, gpu, world_size=8, num_nodes=2, tp_size=8).tp_all_reduce(64)
        self.assertAlmostEqual(inter / intra, gpu.nvlink_bw / gpu.rdma_bw)

    def test_kv_heads_replicate_instead_of_zero(self):
        # Bug 5: num_kv_heads // tp_size floored to 0.
        c = cfg("qwen3-30B-A3B_config.json")  # 4 KV heads
        self.assertEqual(tp_num_kv_heads(4, 8), 1)
        self.assertGreater(get_kvcache_size(c, False, 8), 0)
        self.assertEqual(get_kvcache_size(c, False, 8), get_kvcache_size(c, False, 4))
        with self.assertRaises(ValueError):
            tp_num_kv_heads(4, 6)

    def test_mla_attention_core_flops_shard_with_tp(self):
        # Bug 6: MLA attention core used all heads regardless of TP.
        c = SimpleNamespace(
            num_attention_heads=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
        )
        tp1 = MLA(c, False, False, 1)
        tp8 = MLA(c, False, False, 8)
        for fn in ("get_attn_core_gflops_absorb", "get_attn_core_gflops_noabsorb"):
            ratio = getattr(tp1, fn)(1, 4096) / getattr(tp8, fn)(1, 4096)
            self.assertAlmostEqual(ratio, 8.0)


class ModellingTest(unittest.TestCase):
    def test_bandwidth_uses_si_bytes(self):
        self.assertAlmostEqual(transfer_time(1e9, 1.0), 1.0)
        self.assertEqual(BYTES_PER_GIB, 1024**3)

    def test_up_down_mfu_is_flop_weighted_harmonic(self):
        self.assertAlmostEqual(combine_up_down_mfu(0.6, 0.6), 0.6)
        mixed = combine_up_down_mfu(0.8, 0.4)
        # time = (2/3)/0.8 + (1/3)/0.4 = 5/3  ->  mfu = 3/5
        self.assertAlmostEqual(mixed, 0.6)
        self.assertLess(mixed, max(0.8, 0.4))

    def test_gdn_params_include_out_proj_and_bf16_conv(self):
        c = cfg("qwen3-next-80B-A3B_config.json")
        key_dim = c.linear_num_key_heads * c.linear_key_head_dim
        value_dim = c.linear_num_value_heads * c.linear_value_head_dim
        h = c.hidden_size
        gemm = 2 * h * key_dim + 2 * h * value_dim + 2 * h * c.linear_num_value_heads
        gemm += value_dim * h  # out_proj
        conv = (2 * key_dim + value_dim) * c.linear_conv_kernel_dim
        self.assertEqual(get_gdn_params_size(c, False, 1), 2 * gemm + 2 * conv)
        self.assertEqual(get_gdn_params_size(c, True, 1), gemm + 2 * conv)

    def test_gdn_params_shard_with_tp(self):
        c = cfg("qwen3.5-122B-A10B_config.json")
        self.assertLess(
            get_gdn_params_size(c, True, 2), get_gdn_params_size(c, True, 1)
        )


class RobustnessTest(unittest.TestCase):
    def test_gemm_lookup_is_independent_of_csv_row_order(self):
        path = os.path.join(ROOT, "bench_data/gemm/h20/data.csv")
        with open(path) as f:
            header, *rows = f.read().splitlines()
        tmp_root = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp_root, "bench_data/gemm/h20"))
        with open(os.path.join(tmp_root, "bench_data/gemm/h20/data.csv"), "w") as f:
            f.write("\n".join([header] + rows[::-1]) + "\n")
        old_root = diagnostics.REPO_ROOT
        try:
            with redirect_stdout(StringIO()):
                ordered = get_gemm_mfu("H20", 512, 4096, 4096)
                diagnostics.REPO_ROOT = tmp_root
                reversed_ = get_gemm_mfu("H20", 512, 4096, 4096)
        finally:
            diagnostics.REPO_ROOT = old_root
        self.assertEqual(ordered, reversed_)

    def test_invalid_inputs_are_rejected(self):
        cases = [
            ["--config-path", "hf_configs/qwen3-8B_config.json", "--decode-bs", "0"],
            [
                "--config-path",
                "hf_configs/qwen3-8B_config.json",
                "--world-size",
                "3",
                "--tp-size",
                "2",
            ],
            [
                "--config-path",
                "hf_configs/qwen3-30B-A3B_config.json",
                "--world-size",
                "3",
            ],
            [
                "--config-path",
                "hf_configs/qwen3-8B_config.json",
                "--prefill-only",
                "--decode-only",
            ],
        ]
        for case in cases:
            with self.subTest(case):
                self.assertNotEqual(run_main(*case).returncode, 0)

    def test_strict_mode_fails_on_fallbacks(self):
        base = [
            "--config-path",
            "hf_configs/qwen3-8B_config.json",
            "--device-type",
            "H200",
            "--decode-bs",
            "64",
            "--decode-only",
        ]
        self.assertEqual(run_main(*base).returncode, 0)  # H200 has no bench data
        self.assertEqual(run_main(*base, "--strict").returncode, 1)

    def test_runs_from_any_working_directory(self):
        res = subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "main.py"),
                "--config-path",
                os.path.join(ROOT, "hf_configs/qwen3-8B_config.json"),
                "--decode-bs",
                "64",
                "--decode-only",
            ],
            capture_output=True,
            text=True,
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("Warnings / fallbacks used:               0", res.stdout)


if __name__ == "__main__":
    unittest.main()
