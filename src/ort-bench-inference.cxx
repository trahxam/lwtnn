// ort-bench-inference.cxx
//
// Benchmarks ONNX Runtime C++ inference: reports per-call timing statistics
// and peak RSS memory, in the same key=value format as lwtnn-bench-inference.
//
// Usage:
//   ort-bench-inference <model.onnx> [--n-inferences N]
//                                    [--n-inputs I] [--n-timesteps T]
//
// The test input matches the ramp used by lwtnn-bench-inference:
//   normalised_input[j, i] = linspace(-1,1,n_inputs)[i]
//                           * linspace(-1,1,n_timesteps)[j]
//
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ── peak RSS ──────────────────────────────────────────────────────────────────

static long peak_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmPeak:", 0) == 0) {
            long kb = 0;
            std::sscanf(line.c_str(), "VmPeak: %ld kB", &kb);
            return kb;
        }
    }
    return -1;
}

// ── ramp input ────────────────────────────────────────────────────────────────

// Returns a flat float buffer of shape (1, n_timesteps, n_inputs), row-major.
// normalised[j, i] = linspace(-1,1,n_inputs)[i] * linspace(-1,1,n_timesteps)[j]
static std::vector<float> make_ramp_input(int n_inputs, int n_timesteps) {
    std::vector<float> buf(n_inputs * n_timesteps);
    for (int i = 0; i < n_inputs; ++i) {
        double xi = (n_inputs == 1) ? 0.0
                    : (-1.0 + i * 2.0 / (n_inputs - 1));
        for (int j = 0; j < n_timesteps; ++j) {
            double yj = (n_timesteps == 1) ? 0.0
                        : (-1.0 + j * 2.0 / (n_timesteps - 1));
            buf[j * n_inputs + i] = static_cast<float>(xi * yj);
        }
    }
    return buf;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <model.onnx> [--n-inferences N]"
                     " [--n-inputs I] [--n-timesteps T]\n";
        return 1;
    }

    std::string model_path  = argv[1];
    int n_inferences = 1000;
    int n_inputs     = 19;   // GRU network default
    int n_timesteps  = 20;   // matches lwtnn-test-rnn default

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-inferences" && i + 1 < argc) n_inferences = std::stoi(argv[++i]);
        else if (arg == "--n-inputs"     && i + 1 < argc) n_inputs     = std::stoi(argv[++i]);
        else if (arg == "--n-timesteps"  && i + 1 < argc) n_timesteps  = std::stoi(argv[++i]);
    }

    // ── ORT setup ─────────────────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort-bench");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetInterOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_opts);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr  = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name  = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();

    // ── detect input rank from model (2-D for MLP, 3-D for RNN) ─────────────
    auto input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    int input_rank = static_cast<int>(input_tensor_info.GetShape().size());

    // ── build input tensor ────────────────────────────────────────────────────
    std::vector<float> input_data;
    std::vector<int64_t> input_shape;
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    if (input_rank == 2) {
        // MLP: (1, n_inputs)
        input_data.resize(n_inputs);
        for (int i = 0; i < n_inputs; ++i) {
            double xi = (n_inputs == 1) ? 0.0 : (-1.0 + i * 2.0 / (n_inputs - 1));
            input_data[i] = static_cast<float>(xi);
        }
        input_shape = {1, (int64_t)n_inputs};
    } else {
        // RNN: (1, n_timesteps, n_inputs)
        input_data = make_ramp_input(n_inputs, n_timesteps);
        input_shape = {1, (int64_t)n_timesteps, (int64_t)n_inputs};
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    // ── warm-up ───────────────────────────────────────────────────────────────
    {
        const char* in_names[]  = {input_name};
        const char* out_names[] = {output_name};
        auto warm = session.Run(Ort::RunOptions{nullptr},
                                in_names,  &input_tensor, 1,
                                out_names, 1);
    }

    // ── timed loop ────────────────────────────────────────────────────────────
    using Clock = std::chrono::steady_clock;
    using us    = std::chrono::duration<double, std::micro>;

    const char* in_names[]  = {input_name};
    const char* out_names[] = {output_name};

    std::vector<double> times_us;
    times_us.reserve(n_inferences);

    for (int i = 0; i < n_inferences; ++i) {
        auto t0  = Clock::now();
        auto out = session.Run(Ort::RunOptions{nullptr},
                               in_names,  &input_tensor, 1,
                               out_names, 1);
        auto t1  = Clock::now();
        times_us.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
        (void)out;
    }

    // ── stats ─────────────────────────────────────────────────────────────────
    double sum  = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    double mean = sum / n_inferences;
    double min  = *std::min_element(times_us.begin(), times_us.end());
    double max  = *std::max_element(times_us.begin(), times_us.end());

    std::vector<double> sorted = times_us;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[n_inferences / 2];
    double p99    = sorted[static_cast<std::size_t>(0.99 * n_inferences)];

    long rss = peak_rss_kb();

    // ── final inference for output verification ───────────────────────────────
    auto final_out_vec = session.Run(Ort::RunOptions{nullptr},
                                     in_names,  &input_tensor, 1,
                                     out_names, 1);
    float* probs = final_out_vec[0].GetTensorMutableData<float>();
    auto out_shape = final_out_vec[0].GetTensorTypeAndShapeInfo().GetShape();
    int n_classes = static_cast<int>(out_shape.back());

    // ── output ────────────────────────────────────────────────────────────────
    std::cout << "backend=onnxruntime\n";
    std::cout << "n_inferences=" << n_inferences << "\n";
    std::printf("inference_mean_us=%.3f\n",   mean);
    std::printf("inference_min_us=%.3f\n",    min);
    std::printf("inference_median_us=%.3f\n", median);
    std::printf("inference_p99_us=%.3f\n",    p99);
    std::printf("inference_max_us=%.3f\n",    max);
    std::printf("peak_rss_kb=%ld\n",          rss);
    std::cout << "outputs=";
    for (int i = 0; i < n_classes; ++i) {
        if (i > 0) std::cout << " ";
        std::printf("class_%d=%.6f", i, probs[i]);
    }
    std::cout << "\n";

    return 0;
}
