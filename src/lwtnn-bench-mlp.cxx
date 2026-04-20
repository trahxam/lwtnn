// lwtnn-bench-mlp.cxx
//
// Benchmarks lwtnn feedforward (MLP) inference via LightweightNeuralNetwork.
// Reports per-call timing statistics and peak RSS in the same key=value
// format as lwtnn-bench-inference.
//
// Usage:
//   lwtnn-bench-mlp <nn_config.json> [--n-inferences N]
//
// Inputs are generated with the 1-D ramp from test_utilities.hh:
//   normalised[i] = linspace(-1, 1, n_inputs)[i]
//
#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include "test_utilities.hh"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <nn_config.json> [--n-inferences N]\n";
        return 1;
    }

    std::string config_path = argv[1];
    int n_inferences = 1000;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--n-inferences" && i + 1 < argc)
            n_inferences = std::stoi(argv[++i]);
    }

    // ── load model ────────────────────────────────────────────────────────────
    std::ifstream config_file(config_path);
    if (!config_file) {
        std::cerr << "cannot open " << config_path << "\n";
        return 1;
    }
    auto config = lwt::parse_json(config_file);
    lwt::LightweightNeuralNetwork net(
        config.inputs, config.layers, config.outputs);

    // ── build test inputs: 1-D ramp (matches lwtnn-test-arbitrary-net) ────────
    const std::size_t n_inputs = config.inputs.size();
    std::map<std::string, double> inputs;
    for (std::size_t i = 0; i < n_inputs; ++i) {
        const auto& inp = config.inputs.at(i);
        inputs[inp.name] = ramp(inp, i, n_inputs);
    }

    // ── warm-up ───────────────────────────────────────────────────────────────
    auto warm = net.compute(inputs);

    // ── timed loop ────────────────────────────────────────────────────────────
    using Clock = std::chrono::steady_clock;
    using us    = std::chrono::duration<double, std::micro>;

    std::vector<double> times_us;
    times_us.reserve(n_inferences);

    for (int i = 0; i < n_inferences; ++i) {
        auto t0  = Clock::now();
        auto out = net.compute(inputs);
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

    // ── verify output ─────────────────────────────────────────────────────────
    auto final_out = net.compute(inputs);

    // ── key=value output ──────────────────────────────────────────────────────
    std::cout << "backend=lwtnn\n";
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
