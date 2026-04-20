// lwtnn-bench-inference.cxx
//
// Benchmarks lwtnn RNN inference: reports per-call timing statistics and
// peak RSS memory.  Outputs key=value lines so the Python harness can parse
// them without fragile column-parsing.
//
// Usage:
//   lwtnn-bench-inference <nn_config.json> [--n-inferences N]
//
// The test inputs replicate lwtnn-test-rnn's default "ramp" pattern:
//   after normalisation, input i at time-step j =
//       linspace(-1,1,n_inputs)[i] * linspace(-1,1,n_patterns)[j]
//
#include "lwtnn/generic/LightweightNeuralNetwork.hh"
#include "lwtnn/generic/LightweightNeuralNetwork.tcc"
#include "lwtnn/parse_json.hh"
#include "test_utilities.hh"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

// ── peak RSS from /proc/self/status ──────────────────────────────────────────

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

// ── helpers ───────────────────────────────────────────────────────────────────

static void usage(const std::string& name) {
    std::cerr << "usage: " << name
              << " <nn_config.json> [--n-inferences N]\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string config_path = argv[1];
    int n_inferences = 1000;

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--n-inferences" && i + 1 < argc) {
            n_inferences = std::stoi(argv[++i]);
        }
    }

    // ── load model ────────────────────────────────────────────────────────────
    std::ifstream config_file(config_path);
    if (!config_file) {
        std::cerr << "cannot open " << config_path << "\n";
        return 1;
    }
    auto config = lwt::parse_json(config_file);

    lwt::generic::LightweightRNN<float> rnn(config.inputs, config.layers, config.outputs);

    // ── build test inputs (same ramp as lwtnn-test-rnn default) ──────────────
    const std::size_t n_patterns = 20;
    const lwt::VectorMap inputs = get_values_vec(config.inputs, n_patterns);

    // ── warm-up ───────────────────────────────────────────────────────────────
    auto warm = rnn.reduce(inputs);

    // ── timed inference loop ──────────────────────────────────────────────────
    using Clock = std::chrono::steady_clock;
    using us    = std::chrono::duration<double, std::micro>;

    std::vector<double> times_us;
    times_us.reserve(n_inferences);

    for (int i = 0; i < n_inferences; ++i) {
        auto t0  = Clock::now();
        auto out = rnn.reduce(inputs);
        auto t1  = Clock::now();
        times_us.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
        (void)out;
    }

    // ── compute stats ─────────────────────────────────────────────────────────
    double sum  = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    double mean = sum / n_inferences;
    double min  = *std::min_element(times_us.begin(), times_us.end());
    double max  = *std::max_element(times_us.begin(), times_us.end());

    // median
    std::vector<double> sorted = times_us;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[n_inferences / 2];

    // p99
    double p99 = sorted[static_cast<std::size_t>(0.99 * n_inferences)];

    long rss = peak_rss_kb();

    // ── verify output ─────────────────────────────────────────────────────────
    auto final_out = rnn.reduce(inputs);

    // ── print results (key=value, one per line) ───────────────────────────────
    std::cout << "backend=lwtnn-f32\n";
    std::cout << "n_inferences=" << n_inferences << "\n";
    std::printf("inference_mean_us=%.3f\n",   mean);
    std::printf("inference_min_us=%.3f\n",    min);
    std::printf("inference_median_us=%.3f\n", median);
    std::printf("inference_p99_us=%.3f\n",    p99);
    std::printf("inference_max_us=%.3f\n",    max);
    std::printf("peak_rss_kb=%ld\n",          rss);
    std::cout << "outputs=";
    bool first = true;
    for (const auto& kv : final_out) {
        if (!first) std::cout << " ";
        std::printf("%s=%.6f", kv.first.c_str(), kv.second);
        first = false;
    }
    std::cout << "\n";

    return 0;
}
