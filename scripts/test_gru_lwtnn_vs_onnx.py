#!/usr/bin/env python3
"""
Compare lwtnn vs ONNX inference on the GRU test network.

Downloads the GRU test model used in lwtnn's test suite, converts it to
both the lwtnn JSON format and an ONNX model (via PyTorch), runs inference
with both, and compares the outputs and benchmarks performance.

The network architecture (Keras 1.x / Theano backend):
  Masking → GRU(25) → MaxoutDense(64, 5 pieces) → Dense(64) → Highway(64) → Dense(4) → Softmax

Run from the lwtnn repo root:
  python scripts/test_gru_lwtnn_vs_onnx.py [--n-inferences N]

Requirements (all available in the pixi env):
  h5py, torch, onnxruntime
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT         = Path(__file__).resolve().parent.parent
LWTNN_BIN          = REPO_ROOT / "build" / "bin" / "lwtnn-test-rnn"
LWTNN_BENCH_BIN    = REPO_ROOT / "build" / "bin" / "lwtnn-bench-inference"
LWTNN_F32_BENCH_BIN = REPO_ROOT / "build" / "bin" / "lwtnn-bench-inference-f32"
ORT_BENCH_BIN      = REPO_ROOT / "build" / "bin" / "ort-bench-inference"
CONVERTER         = REPO_ROOT / "converters" / "keras2json.py"

TEST_DATA_URL = "https://github.com/lwtnn/lwtnn-test-data/raw/v2.1/GRU.tgz"
N_PATTERNS    = 20   # number of time-steps used by lwtnn-test-rnn

# ── hard sigmoid (Theano convention) ─────────────────────────────────────────

def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """clamp(0.2*x + 0.5, 0, 1)  — matches theano/lwtnn implementation."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

# ── model layers ─────────────────────────────────────────────────────────────

class TheanoGRU(nn.Module):
    """
    GRU matching old Keras 1.x (Theano backend) and lwtnn conventions:
      z = hard_sigmoid(x @ W_z + h @ U_z + b_z)
      r = hard_sigmoid(x @ W_r + h @ U_r + b_r)
      hh = tanh(x @ W_h + (r*h) @ U_h + b_h)
      h_new = z*h + (1-z)*hh

    All weight matrices are in (input_dim, output_dim) / (hidden_dim, hidden_dim)
    convention, matching HDF5 storage (Theano row-major convention).
    """

    def __init__(self, W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h):
        super().__init__()
        for name, arr in [
            ("W_z", W_z), ("W_r", W_r), ("W_h", W_h),
            ("U_z", U_z), ("U_r", U_r), ("U_h", U_h),
            ("b_z", b_z), ("b_r", b_r), ("b_h", b_h),
        ]:
            self.register_buffer(name, torch.tensor(arr, dtype=torch.float32))
        self.hidden_size = W_z.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            xt = x[:, t, :]
            z  = hard_sigmoid(xt @ self.W_z + h @ self.U_z + self.b_z)
            r  = hard_sigmoid(xt @ self.W_r + h @ self.U_r + self.b_r)
            hh = torch.tanh(xt @ self.W_h + (r * h) @ self.U_h + self.b_h)
            h  = z * h + (1 - z) * hh
        return h  # (batch, hidden_size)


class MaxoutDense(nn.Module):
    """
    output = max_k( x @ W[k] + b[k] )
    W: (nb_feature, input_dim, output_dim)
    b: (nb_feature, output_dim)
    """

    def __init__(self, W, b):
        super().__init__()
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        # result per feature: x @ W[k] + b[k]  → (batch, output_dim)
        out = torch.einsum("bi,kio->bko", x, self.W) + self.b  # (batch, nb_feature, out_dim)
        return out.max(dim=1).values  # (batch, out_dim)


class Highway(nn.Module):
    """
    t   = sigmoid(x @ W_carry + b_carry)   (transform gate)
    h   = relu(x @ W + b)
    out = t * h + (1 - t) * x
    """

    def __init__(self, W, b, W_carry, b_carry):
        super().__init__()
        for name, arr in [("W", W), ("b", b), ("W_carry", W_carry), ("b_carry", b_carry)]:
            self.register_buffer(name, torch.tensor(arr, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(x @ self.W_carry + self.b_carry)
        h = torch.relu(x @ self.W + self.b)
        return t * h + (1 - t) * x


class LinearLayer(nn.Module):
    """Dense layer with linear activation, weights in (input, output) convention."""

    def __init__(self, W, b):
        super().__init__()
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b


class GRUNet(nn.Module):
    """Full GRU network matching the lwtnn test model architecture."""

    def __init__(self, h5_path: str):
        super().__init__()
        with h5py.File(h5_path, "r") as f:
            def w(key):
                return np.array(f[key])

            self.gru = TheanoGRU(
                W_z=w("gru_1/gru_1_W_z"), W_r=w("gru_1/gru_1_W_r"), W_h=w("gru_1/gru_1_W_h"),
                U_z=w("gru_1/gru_1_U_z"), U_r=w("gru_1/gru_1_U_r"), U_h=w("gru_1/gru_1_U_h"),
                b_z=w("gru_1/gru_1_b_z"), b_r=w("gru_1/gru_1_b_r"), b_h=w("gru_1/gru_1_b_h"),
            )
            self.maxout  = MaxoutDense(w("maxoutdense_1/maxoutdense_1_W"),
                                       w("maxoutdense_1/maxoutdense_1_b"))
            self.dense1  = LinearLayer(w("dense_1/dense_1_W"), w("dense_1/dense_1_b"))
            self.highway = Highway(w("highway_1/highway_1_W"),    w("highway_1/highway_1_b"),
                                   w("highway_1/highway_1_W_carry"), w("highway_1/highway_1_b_carry"))
            self.dense2  = LinearLayer(w("dense_2/dense_2_W"), w("dense_2/dense_2_b"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)  — already normalised
        h = self.gru(x)
        h = self.maxout(h)
        h = self.dense1(h)
        h = self.highway(h)
        h = self.dense2(h)
        return torch.softmax(h, dim=-1)

# ── input generation (mirrors lwtnn test_utilities.cxx) ──────────────────────

def make_normalized_input(n_inputs: int, n_patterns: int) -> torch.Tensor:
    """
    Replicates the 2-D ramp in lwtnn's test_utilities.cxx::get_values_vec.

    After lwtnn's internal normalisation, variable i at time-step j becomes:
        linspace(-1, 1, n_inputs)[i]  *  linspace(-1, 1, n_patterns)[j]

    Returns shape (1, n_patterns, n_inputs).
    """
    x = np.linspace(-1.0, 1.0, n_inputs)    # (n_inputs,)
    y = np.linspace(-1.0, 1.0, n_patterns)  # (n_patterns,)
    grid = x[:, None] * y[None, :]          # (n_inputs, n_patterns)
    # transpose so axis 0 = time, axis 1 = feature
    seq = grid.T                            # (n_patterns, n_inputs)
    return torch.tensor(seq[None], dtype=torch.float32)  # (1, n_patterns, n_inputs)

# ── optimised ONNX model (native GRU operator) ───────────────────────────────

def build_native_gru_onnx(h5_path: str, onnx_path: str) -> None:
    """
    Build an ONNX model that uses the *native* ONNX GRU operator instead of
    the unrolled per-timestep graph that torch.onnx.export produces from a
    Python for-loop.

    ORT has a highly-optimised C++ GRU kernel; by using the ONNX GRU op
    (with HardSigmoid activations to match lwtnn exactly) we replace ~700
    scattered nodes with a single fused kernel call.

    Weight layout required by the ONNX GRU op (opset 17, direction=forward):
      W : (1, 3*hidden, input)   — gates ordered z, r, h
      R : (1, 3*hidden, hidden)  — same gate order
      B : (1, 6*hidden)          — [Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h]
    """
    import onnx
    from onnx import numpy_helper, helper, TensorProto

    with h5py.File(h5_path) as f:
        def w(k):
            return np.array(f[k], dtype=np.float32)

        W_z, W_r, W_h = w("gru_1/gru_1_W_z"), w("gru_1/gru_1_W_r"), w("gru_1/gru_1_W_h")
        U_z, U_r, U_h = w("gru_1/gru_1_U_z"), w("gru_1/gru_1_U_r"), w("gru_1/gru_1_U_h")
        b_z, b_r, b_h = w("gru_1/gru_1_b_z"), w("gru_1/gru_1_b_r"), w("gru_1/gru_1_b_h")

        W_mo = w("maxoutdense_1/maxoutdense_1_W")   # (nb_feat=5, in=25, out=64)
        b_mo = w("maxoutdense_1/maxoutdense_1_b")   # (5, 64)
        W_d1 = w("dense_1/dense_1_W")              # (64, 64)
        b_d1 = w("dense_1/dense_1_b")
        W_hw = w("highway_1/highway_1_W")
        b_hw = w("highway_1/highway_1_b")
        W_hc = w("highway_1/highway_1_W_carry")
        b_hc = w("highway_1/highway_1_b_carry")
        W_d2 = w("dense_2/dense_2_W")              # (64, 4)
        b_d2 = w("dense_2/dense_2_b")

    hidden   = W_z.shape[1]          # 25
    n_in     = W_z.shape[0]          # 19
    nb_feat  = W_mo.shape[0]         # 5   (maxout pieces)
    nb_out   = W_mo.shape[2]         # 64  (maxout output dim)

    # ── GRU weights ────────────────────────────────────────────────────────────
    # HDF5 stores W as (input, hidden); ONNX wants (hidden, input) → transpose
    W_gru = np.concatenate([W_z.T, W_r.T, W_h.T], axis=0)[None]          # (1, 3h, in)
    R_gru = np.concatenate([U_z.T, U_r.T, U_h.T], axis=0)[None]          # (1, 3h, h)
    B_gru = np.concatenate([b_z, b_r, b_h,
                             np.zeros(3 * hidden, dtype=np.float32)])[None]  # (1, 6h)

    # ── MaxoutDense weight pre-processing ──────────────────────────────────────
    # W_mo is (nb_feat=5, in=25, out=64).  Avoid batched-matmul broadcast
    # ambiguity by pre-folding into a 2-D weight:
    #   transpose (5,25,64) → (25,5,64), then reshape → (25, 5*64=320)
    # so at runtime:  h0 (1,25) @ W_mo_flat (25,320) = (1,320)
    #   → reshape (1,320) → (1,5,64)  → add b_mo (5,64)  → ReduceMax axis=1
    W_mo_flat = W_mo.transpose(1, 0, 2).reshape(hidden, nb_feat * nb_out).astype(np.float32)

    # ── initializers (weights as named tensors in the graph) ──────────────────
    def init(name, arr):
        return numpy_helper.from_array(np.ascontiguousarray(arr), name=name)

    inits = [
        init("gru_W",  W_gru), init("gru_R",  R_gru), init("gru_B",  B_gru),
        init("mo_W",   W_mo_flat),
        init("mo_b",   b_mo),
        init("d1_W",   W_d1),  init("d1_b",   b_d1),
        init("hw_W",   W_hw),  init("hw_b",   b_hw),
        init("hw_Wc",  W_hc),  init("hw_bc",  b_hc),
        init("d2_W",   W_d2),  init("d2_b",   b_d2),
        # scalar constant
        init("ones_1",       np.ones(1,  dtype=np.float32)),
        # shape tensors for Squeeze / Unsqueeze
        init("sq_axes",      np.array([0],                          dtype=np.int64)),
        # shape tensor for Reshape after maxout matmul
        init("mo_out_shape", np.array([1, nb_feat, nb_out],         dtype=np.int64)),
    ]

    # ── graph nodes ────────────────────────────────────────────────────────────
    N = helper.make_node   # shorthand

    nodes = [
        # Input is (batch=1, seq, n_in).  ONNX GRU expects (seq, batch, n_in).
        N("Transpose", ["x"],           ["x_seq"],    perm=[1, 0, 2]),

        # ── GRU ── outputs: Y (all seq, unused), Y_h (final hidden)
        helper.make_node(
            "GRU",
            inputs=["x_seq", "gru_W", "gru_R", "gru_B"],
            outputs=["", "gru_Yh"],          # skip Y, keep Y_h
            hidden_size=hidden,
            direction="forward",
            activations=["HardSigmoid", "Tanh"],
            activation_alpha=[0.2, 0.0],     # hard_sigmoid alpha; tanh unused
            activation_beta=[0.5, 0.0],      # hard_sigmoid beta;  tanh unused
        ),

        # Y_h: (num_dir=1, batch, hidden) → squeeze dim 0 → (batch, hidden)
        N("Squeeze",   ["gru_Yh", "sq_axes"], ["h0"]),

        # ── MaxoutDense ─────────────────────────────────────────────────────────
        # W_mo_flat = (hidden=25, nb_feat*nb_out=320) — pre-folded above.
        # h0 (1,25) @ W_flat (25,320) = (1,320) — plain 2-D matmul, no broadcast.
        # Reshape → (1, nb_feat=5, nb_out=64), add bias, take max over piece dim.
        N("MatMul",    ["h0",     "mo_W"],       ["mo_mm"]),
        N("Reshape",   ["mo_mm",  "mo_out_shape"],["mo_rs"]),
        N("Add",       ["mo_rs",  "mo_b"],        ["mo_add"]),
        N("ReduceMax", ["mo_add"],                ["h1"],       axes=[1], keepdims=0),

        # ── Dense 1 (linear) ────────────────────────────────────────────────────
        N("MatMul",  ["h1",    "d1_W"],   ["d1_mm"]),
        N("Add",     ["d1_mm", "d1_b"],   ["h2"]),

        # ── Highway ─────────────────────────────────────────────────────────────
        #   t     = sigmoid(h2 @ W_carry + b_carry)
        #   h_new = t * relu(h2 @ W_hw + b_hw) + (1-t) * h2
        N("MatMul",  ["h2",    "hw_Wc"],  ["hc_mm"]),
        N("Add",     ["hc_mm", "hw_bc"],  ["hc_pre"]),
        N("Sigmoid", ["hc_pre"],          ["t"]),
        N("MatMul",  ["h2",    "hw_W"],   ["hw_mm"]),
        N("Add",     ["hw_mm", "hw_b"],   ["hw_pre"]),
        N("Relu",    ["hw_pre"],          ["h_xform"]),
        N("Mul",     ["t",     "h_xform"],["carry"]),
        N("Sub",     ["ones_1", "t"],     ["one_mt"]),
        N("Mul",     ["one_mt", "h2"],    ["pass"]),
        N("Add",     ["carry",  "pass"],  ["h3"]),

        # ── Dense 2 + Softmax ────────────────────────────────────────────────────
        N("MatMul",   ["h3", "d2_W"],    ["d2_mm"]),
        N("Add",      ["d2_mm", "d2_b"], ["logits"]),
        N("Softmax",  ["logits"],        ["class_probs"], axis=-1),
    ]

    x_info   = helper.make_tensor_value_info("x",           TensorProto.FLOAT, [1, None, n_in])
    out_info = helper.make_tensor_value_info("class_probs", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(nodes, "gru_native_op", [x_info], [out_info], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


# ── C++ runner helpers ────────────────────────────────────────────────────────

def _require_bin(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"binary not found: {path}\n{hint}"
        )

_BUILD_HINT = (
    "Build first:\n"
    "  cmake -DBUILTIN_EIGEN=true -DBUILTIN_BOOST=true -S . -B build\n"
    "  cmake --build build"
)


def run_lwtnn(lwtnn_json: str) -> dict:
    """Run lwtnn-test-rnn and return {label: value}."""
    _require_bin(LWTNN_BIN, _BUILD_HINT)
    result = subprocess.run(
        [str(LWTNN_BIN), lwtnn_json],
        capture_output=True, text=True, check=True,
    )
    return {k: float(v) for k, v in (line.split() for line in result.stdout.strip().splitlines())}


def parse_bench_output(text: str) -> dict:
    """Parse key=value lines from a benchmark binary into a dict."""
    result = {}
    for line in text.strip().splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip()
    return result


def run_lwtnn_bench(lwtnn_json: str, n_inferences: int, f32: bool = False) -> dict:
    """Run lwtnn-bench-inference (f64 or f32) and return the parsed key=value dict."""
    bin_path = LWTNN_F32_BENCH_BIN if f32 else LWTNN_BENCH_BIN
    _require_bin(bin_path, _BUILD_HINT)
    result = subprocess.run(
        [str(bin_path), lwtnn_json, "--n-inferences", str(n_inferences)],
        capture_output=True, text=True, check=True,
    )
    return parse_bench_output(result.stdout)


def run_ort_bench(onnx_path: str, n_inferences: int,
                  n_inputs: int, n_timesteps: int) -> dict:
    """Run ort-bench-inference and return the parsed key=value dict."""
    _require_bin(ORT_BENCH_BIN, _BUILD_HINT)
    result = subprocess.run(
        [str(ORT_BENCH_BIN), onnx_path,
         "--n-inferences", str(n_inferences),
         "--n-inputs",     str(n_inputs),
         "--n-timesteps",  str(n_timesteps)],
        capture_output=True, text=True, check=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ort-bench-inference failed:\n{result.stderr}")
    return parse_bench_output(result.stdout)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-inferences", type=int, default=1000,
                        help="number of inference calls for the benchmark (default: 1000)")
    args = parser.parse_args()
    n_inferences = args.n_inferences

    print("=" * 60)
    print("lwtnn vs ONNX inference comparison — GRU test model")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── 1. download & unpack test data ────────────────────────────────────
        print("\n[1] Downloading test data …")
        tgz_path = tmpdir / "GRU.tgz"
        urllib.request.urlretrieve(TEST_DATA_URL, tgz_path)
        with tarfile.open(tgz_path) as tar:
            tar.extractall(tmpdir)
        arch_json  = tmpdir / "GRU.json"
        variables  = tmpdir / "variables.json"
        h5_weights = tmpdir / "weights" / "GRU_weights.h5"
        print("    done.")

        with open(variables) as f:
            var_cfg = json.load(f)
        class_labels = var_cfg["class_labels"]   # output ordering from Keras
        n_inputs = len(var_cfg["inputs"])

        # ── 2. convert to lwtnn JSON ──────────────────────────────────────────
        print("\n[2] Converting Keras model → lwtnn JSON …")
        lwtnn_json = str(tmpdir / "lwtnn_model.json")
        with open(lwtnn_json, "w") as out_f:
            subprocess.run(
                [sys.executable, str(CONVERTER),
                 str(arch_json), str(variables), str(h5_weights)],
                stdout=out_f, stderr=subprocess.DEVNULL, check=True,
            )
        print("    done.")

        # ── 3. lwtnn inference (correctness) ─────────────────────────────────
        print("\n[3] Running lwtnn inference …")
        lwtnn_out = run_lwtnn(lwtnn_json)
        print("    lwtnn outputs:")
        for label, val in sorted(lwtnn_out.items()):
            print(f"      {label:6s}  {val:.6f}")

        # ── 4. build PyTorch model & export to ONNX ──────────────────────────
        print("\n[4] Building PyTorch model from HDF5 weights …")
        model = GRUNet(str(h5_weights)).eval()
        x = make_normalized_input(n_inputs, N_PATTERNS)  # (1, 20, 19)

        with torch.no_grad():
            torch_probs = model(x).squeeze(0).numpy()
        torch_out = dict(zip(class_labels, torch_probs))
        print("    PyTorch outputs:")
        for label, val in sorted(torch_out.items()):
            print(f"      {label:6s}  {val:.6f}")

        print("\n[5] Exporting to ONNX …")
        onnx_path = str(tmpdir / "gru_model.onnx")
        torch.onnx.export(
            model, x, onnx_path,
            input_names=["normalized_input"],
            output_names=["class_probs"],
            dynamic_axes={"normalized_input": {0: "batch", 1: "seq_len"}},
            opset_version=17,
        )
        print(f"    saved to {onnx_path}")

        # ── 6. ONNX Runtime inference (correctness) ───────────────────────────
        print("\n[6] Running ONNX Runtime inference …")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_probs = sess.run(
            ["class_probs"], {"normalized_input": x.numpy()}
        )[0].squeeze(0)
        onnx_out = dict(zip(class_labels, onnx_probs))
        print("    ONNX outputs:")
        for label, val in sorted(onnx_out.items()):
            print(f"      {label:6s}  {val:.6f}")

        # ── 7. correctness comparison ─────────────────────────────────────────
        print("\n[7] Output correctness comparison")
        print(f"  {'label':6s}  {'lwtnn':>10s}  {'PyTorch':>10s}  {'ONNX':>10s}  "
              f"{'|lwtnn-PT|':>12s}  {'|lwtnn-OX|':>12s}")
        print("  " + "-" * 70)
        all_ok = True
        tol = 1e-5
        for label in sorted(lwtnn_out):
            lv = lwtnn_out[label]
            pv = float(torch_out[label])
            ov = float(onnx_out[label])
            diff_pt = abs(lv - pv)
            diff_ox = abs(lv - ov)
            flag = "" if (diff_pt < tol and diff_ox < tol) else " ← MISMATCH"
            if flag:
                all_ok = False
            print(f"  {label:6s}  {lv:10.6f}  {pv:10.6f}  {ov:10.6f}  "
                  f"{diff_pt:12.2e}  {diff_ox:12.2e}{flag}")
        print()
        if all_ok:
            print(f"  All outputs agree within {tol:.0e}.")
        else:
            print(f"  WARNING: some outputs differ by more than {tol:.0e}.")
            sys.exit(1)

        # ── 8. build optimised ONNX (native GRU op) ──────────────────────────
        print("\n[8] Building optimised ONNX model (native GRU operator) …")
        opt_onnx_path = str(tmpdir / "gru_native_op.onnx")
        build_native_gru_onnx(str(h5_weights), opt_onnx_path)

        # verify it gives the same outputs
        sess_opt = ort.InferenceSession(opt_onnx_path,
                                        providers=["CPUExecutionProvider"])
        opt_probs = sess_opt.run(["class_probs"],
                                 {"x": x.numpy()})[0].squeeze(0)
        opt_out = dict(zip(class_labels, opt_probs))
        print("    Optimised ONNX outputs:")
        for label, val in sorted(opt_out.items()):
            print(f"      {label:6s}  {val:.6f}")

        tol = 1e-5
        mismatches = [l for l in lwtnn_out
                      if abs(lwtnn_out[l] - float(opt_out[l])) > tol]
        if mismatches:
            print(f"  WARNING: mismatch vs lwtnn on: {mismatches}")
        else:
            print(f"    Outputs match lwtnn within {tol:.0e}.")

        import onnx as _onnx
        orig_nodes = len(_onnx.load(onnx_path).graph.node)
        opt_nodes  = len(_onnx.load(opt_onnx_path).graph.node)
        print(f"    Graph size: {orig_nodes} nodes (unrolled)  →  {opt_nodes} nodes (native GRU op)")

        # ── 9. C++ benchmarks ─────────────────────────────────────────────────
        print(f"\n[9] C++ inference benchmarks  ({n_inferences} calls each) …")

        print("    Running lwtnn-bench-inference (f64) …", flush=True)
        lwtnn_bench = run_lwtnn_bench(lwtnn_json, n_inferences, f32=False)

        print("    Running lwtnn-bench-inference (f32) …", flush=True)
        lwtnn_f32_bench = run_lwtnn_bench(lwtnn_json, n_inferences, f32=True)

        print("    Running ort-bench-inference (unrolled) …", flush=True)
        ort_bench = run_ort_bench(onnx_path, n_inferences, n_inputs, N_PATTERNS)

        print("    Running ort-bench-inference (native GRU op) …", flush=True)
        opt_bench = run_ort_bench(opt_onnx_path, n_inferences, n_inputs, N_PATTERNS)

        # ── 10. benchmark comparison table ───────────────────────────────────
        print("\n[10] Performance comparison")
        _print_bench_table(lwtnn_bench, lwtnn_f32_bench, ort_bench, opt_bench)


def _print_bench_table(lwtnn: dict, lwtnn_f32: dict,
                       ort_unrolled: dict, ort_native: dict) -> None:
    """Render a four-way benchmark comparison table."""
    metrics = [
        ("inference_mean_us",   "Mean latency (µs)"),
        ("inference_min_us",    "Min  latency (µs)"),
        ("inference_median_us", "Median latency (µs)"),
        ("inference_p99_us",    "p99  latency (µs)"),
        ("inference_max_us",    "Max  latency (µs)"),
        ("peak_rss_kb",         "Peak RSS (KB)"),
    ]

    col_w   = 13
    label_w = 22
    print(f"  {'Metric':<{label_w}}  {'lwtnn (f64)':>{col_w}}  {'lwtnn (f32)':>{col_w}}  "
          f"{'ORT unrolled':>{col_w}}  {'ORT native':>{col_w}}  "
          f"{'f32/f64':>{8}}  {'native/f64':>{10}}")
    print("  " + "-" * (label_w + 4*col_w + 4*2 + 22))

    for key, label in metrics:
        lv   = float(lwtnn.get(key,       "nan"))
        lv32 = float(lwtnn_f32.get(key,   "nan"))
        ov   = float(ort_unrolled.get(key, "nan"))
        nv   = float(ort_native.get(key,   "nan"))
        r32  = lv32 / lv if lv else float("nan")
        rnat = nv   / lv if lv else float("nan")
        print(f"  {label:<{label_w}}  {lv:>{col_w}.2f}  {lv32:>{col_w}.2f}  "
              f"{ov:>{col_w}.2f}  {nv:>{col_w}.2f}  "
              f"{r32:>{8}.2f}x  {rnat:>{10}.2f}x")

    print()
    mean_l   = float(lwtnn.get("inference_mean_us",     "nan"))
    mean_l32 = float(lwtnn_f32.get("inference_mean_us", "nan"))
    mean_ou  = float(ort_unrolled.get("inference_mean_us", "nan"))
    mean_on  = float(ort_native.get("inference_mean_us",   "nan"))
    rss_l    = float(lwtnn.get("peak_rss_kb", "nan"))
    rss_on   = float(ort_native.get("peak_rss_kb", "nan"))

    print(f"  → lwtnn f32 is {mean_l/mean_l32:.1f}x faster than lwtnn f64.")
    print(f"  → ORT native GRU is {mean_ou/mean_on:.1f}x faster than ORT unrolled.")
    if mean_on < mean_l32:
        print(f"  → ORT native GRU is {mean_l32/mean_on:.1f}x faster than lwtnn f32.")
    else:
        print(f"  → lwtnn f32 is {mean_on/mean_l32:.1f}x faster than ORT native GRU.")
    print(f"  → lwtnn uses {rss_on/rss_l:.1f}x less peak RSS than ORT native GRU.")


if __name__ == "__main__":
    main()
