#!/usr/bin/env python3
"""
test_mlp_lwtnn_vs_onnx.py
=========================
Builds a random MLP (60→128→64→32→16→8, tanh), exports it to:
  - lwtnn JSON  (for C++ lwtnn-bench-mlp)
  - ONNX model  (for C++ ort-bench-inference)

Then verifies numerical equivalence between all three backends and prints a
timing comparison table.

Usage:
    python scripts/test_mlp_lwtnn_vs_onnx.py [--n-inferences N]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# ── architecture ───────────────────────────────────────────────────────────────

LAYER_SIZES = [60, 128, 64, 32, 16, 8]   # input_size, hidden..., output_size
ACTIVATION  = "tanh"


# ── 1. Build and initialise the MLP ──────────────────────────────────────────

def build_mlp(seed=42):
    torch.manual_seed(seed)
    layers = []
    for in_sz, out_sz in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]):
        layers.append(nn.Linear(in_sz, out_sz))
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ── 2. Build the normalised ramp input (matches lwtnn-bench-mlp convention) ──

def make_ramp_input(n_inputs):
    """linspace(-1, 1, n_inputs) — same as the C++ ramp() helper."""
    return np.linspace(-1.0, 1.0, n_inputs, dtype=np.float64)


# ── 3. Run lwtnn C++ inference ────────────────────────────────────────────────

def write_lwtnn_json(model, path):
    """Serialise the MLP to the lwtnn sequential JSON format."""
    n_inputs = LAYER_SIZES[0]

    # inputs: normalise from raw ramp → (-1, 1)
    # raw value from linspace(-1,1) already IS the normalised value, so
    # offset = 0, scale = 1 leaves it unchanged: (raw + 0) * 1 = raw.
    inputs_json = [
        {"name": f"in_{i}", "offset": 0.0, "scale": 1.0}
        for i in range(n_inputs)
    ]

    # layers
    layers_json = []
    linear_idx = 0
    for module in model:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy()   # (out, in)
            b = module.bias.detach().numpy()     # (out,)
            # lwtnn matrix convention: matrix(row, col) = weights[col + row*n_in]
            # → W.flatten() (row-major C order) matches directly
            layers_json.append({
                "architecture": "dense",
                "activation":   ACTIVATION,
                "weights":      W.flatten().tolist(),
                "bias":         b.tolist(),
            })
            linear_idx += 1

    n_outputs = LAYER_SIZES[-1]
    outputs_json = [f"out_{i}" for i in range(n_outputs)]

    config = {"inputs": inputs_json, "layers": layers_json, "outputs": outputs_json}
    with open(path, "w") as f:
        json.dump(config, f)


def run_lwtnn(model, bench_bin, n_inferences=1000):
    """Write lwtnn JSON, call the C++ binary, return (outputs_dict, bench_dict)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json_path = f.name
    try:
        write_lwtnn_json(model, json_path)
        result = subprocess.run(
            [bench_bin, json_path, "--n-inferences", str(n_inferences)],
            capture_output=True, text=True, check=True,
        )
    finally:
        os.unlink(json_path)

    kv = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()

    outputs = {}
    if "outputs" in kv:
        for pair in kv["outputs"].split():
            k, _, v = pair.partition("=")
            outputs[k] = float(v)

    return outputs, kv


# ── 4. Build ONNX model (using PyTorch export) ────────────────────────────────

def build_onnx_model(model, path):
    """Export the MLP to ONNX via torch.onnx.export."""
    n_inputs = LAYER_SIZES[0]
    dummy = torch.zeros(1, n_inputs, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


# ── 5. ONNX Runtime Python inference ─────────────────────────────────────────

def run_ort_python(model, onnx_path):
    """Run one forward pass through ORT (Python) and return output array."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    n_inputs = LAYER_SIZES[0]
    ramp = make_ramp_input(n_inputs).astype(np.float32)[None]   # (1, 60)
    out = sess.run(None, {"input": ramp})[0]  # (1, 8)
    return out[0]


# ── 6. PyTorch reference forward pass ─────────────────────────────────────────

def run_pytorch(model):
    n_inputs = LAYER_SIZES[0]
    ramp = torch.tensor(make_ramp_input(n_inputs), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(ramp)
    return out.squeeze(0).numpy()


# ── 7. C++ ORT benchmark ──────────────────────────────────────────────────────

def run_ort_bench(onnx_path, bench_bin, n_inferences=1000, n_inputs=60):
    result = subprocess.run(
        [bench_bin, onnx_path,
         "--n-inferences", str(n_inferences),
         "--n-inputs", str(n_inputs)],
        capture_output=True, text=True, check=True,
    )
    kv = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    outputs = {}
    if "outputs" in kv:
        for pair in kv["outputs"].split():
            k, _, v = pair.partition("=")
            outputs[k] = float(v)
    return outputs, kv


# ── 8. Comparison table ───────────────────────────────────────────────────────

def _print_bench_table(lwtnn_kv, lwtnn_f32_kv, ort_kv):
    metrics = [
        ("mean",          "inference_mean_us"),
        ("min",           "inference_min_us"),
        ("median",        "inference_median_us"),
        ("p99",           "inference_p99_us"),
        ("max",           "inference_max_us"),
        ("peak RSS (kB)", "peak_rss_kb"),
    ]
    col = 14
    print(f"\n{'Metric':<20}  {'lwtnn (f64)':>{col}}  {'lwtnn (f32)':>{col}}  {'ORT (f32)':>{col}}")
    print("-" * (20 + 3*(col+2)))
    for label, key in metrics:
        lv   = lwtnn_kv.get(key,     "N/A")
        lv32 = lwtnn_f32_kv.get(key, "N/A")
        ov   = ort_kv.get(key,       "N/A")
        try:
            r32  = float(lv32) / float(lv)
            rort = float(ov)   / float(lv)
            ratio_str = f"  ({r32:.2f}×)  ({rort:.2f}×)"
        except (ValueError, ZeroDivisionError):
            ratio_str = ""
        print(f"{label:<20}  {lv:>{col}}  {lv32:>{col}}  {ov:>{col}}{ratio_str}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-inferences", type=int, default=1000)
    parser.add_argument("--build-dir", default=None,
                        help="lwtnn build directory (default: auto-detect)")
    args = parser.parse_args()

    # locate build dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir   = os.path.dirname(script_dir)

    if args.build_dir:
        build_dir = args.build_dir
    else:
        candidates = [
            os.path.join(repo_dir, "build"),
            os.path.join(repo_dir, "build_pixi"),
        ]
        build_dir = next((d for d in candidates if os.path.isdir(d)), None)
        if build_dir is None:
            sys.exit("Cannot find build dir; pass --build-dir")

    lwtnn_bench     = os.path.join(build_dir, "bin", "lwtnn-bench-mlp")
    lwtnn_f32_bench = os.path.join(build_dir, "bin", "lwtnn-bench-mlp-f32")
    ort_bench       = os.path.join(build_dir, "bin", "ort-bench-inference")

    for p, name in [(lwtnn_bench,     "lwtnn-bench-mlp"),
                    (lwtnn_f32_bench, "lwtnn-bench-mlp-f32"),
                    (ort_bench,       "ort-bench-inference")]:
        if not os.path.isfile(p):
            sys.exit(f"Binary not found: {p}\nRun cmake --build {build_dir}")

    # ── build model ────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"MLP architecture: {' → '.join(str(s) for s in LAYER_SIZES)}, {ACTIVATION}")
    print("=" * 60)

    model = build_mlp()
    model.eval()

    # ── PyTorch reference ──────────────────────────────────────────────────────
    pt_out = run_pytorch(model)
    print(f"\n[1] PyTorch output: {pt_out}")

    # ── lwtnn C++ (float64) ───────────────────────────────────────────────────
    print(f"\n[2] Running lwtnn C++ float64 ({args.n_inferences} inferences)...")
    lwtnn_out, lwtnn_kv = run_lwtnn(model, lwtnn_bench, args.n_inferences)
    lwtnn_vec = np.array([lwtnn_out[f"out_{i}"] for i in range(LAYER_SIZES[-1])])
    print(f"    lwtnn-f64 output: {lwtnn_vec}")
    diff_pt = np.max(np.abs(lwtnn_vec - pt_out))
    print(f"    max |lwtnn-f64 - pytorch| = {diff_pt:.2e}")
    assert diff_pt < 1e-5, f"lwtnn-f64 vs pytorch mismatch: {diff_pt}"

    # ── lwtnn C++ (float32) ───────────────────────────────────────────────────
    print(f"\n[3] Running lwtnn C++ float32 ({args.n_inferences} inferences)...")
    lwtnn_f32_out, lwtnn_f32_kv = run_lwtnn(model, lwtnn_f32_bench, args.n_inferences)
    lwtnn_f32_vec = np.array([lwtnn_f32_out[f"out_{i}"] for i in range(LAYER_SIZES[-1])])
    print(f"    lwtnn-f32 output: {lwtnn_f32_vec}")
    diff_f32 = np.max(np.abs(lwtnn_f32_vec - pt_out))
    print(f"    max |lwtnn-f32 - pytorch| = {diff_f32:.2e}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        build_onnx_model(model, onnx_path)
        onnx_model = onnx.load(onnx_path)
        n_nodes = len(onnx_model.graph.node)
        print(f"\n[4] ONNX model: {n_nodes} nodes")

        # ORT Python check
        ort_py_out = run_ort_python(model, onnx_path)
        print(f"    ORT (Python) output: {ort_py_out}")
        diff_ort = np.max(np.abs(ort_py_out - pt_out))
        print(f"    max |ORT - pytorch| = {diff_ort:.2e}")
        assert diff_ort < 1e-5, f"ORT vs pytorch mismatch: {diff_ort}"

        # ── ORT C++ benchmark ─────────────────────────────────────────────────
        print(f"\n[5] Running ORT C++ ({args.n_inferences} inferences)...")
        ort_out, ort_kv = run_ort_bench(
            onnx_path, ort_bench, args.n_inferences, n_inputs=LAYER_SIZES[0])
        ort_vec = np.array([ort_out[f"class_{i}"] for i in range(LAYER_SIZES[-1])])
        print(f"    ORT C++ output: {ort_vec}")
        diff_ort_cpp = np.max(np.abs(ort_vec - pt_out))
        print(f"    max |ORT C++ - pytorch| = {diff_ort_cpp:.2e}")
        assert diff_ort_cpp < 1e-5, f"ORT C++ vs pytorch mismatch: {diff_ort_cpp}"

    finally:
        os.unlink(onnx_path)

    # ── timing table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Timing comparison ({args.n_inferences} inferences, µs unless noted)")
    _print_bench_table(lwtnn_kv, lwtnn_f32_kv, ort_kv)
    print()


if __name__ == "__main__":
    main()
