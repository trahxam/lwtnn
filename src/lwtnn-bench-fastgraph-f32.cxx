// lwtnn-bench-fastgraph-f32.cxx
//
// Benchmarks lwtnn FastGraph<float> inference on a graph-format MLP config.
// FastGraph takes pre-ordered Eigen vectors directly (no map lookups), which
// is faster than LightweightNeuralNetwork's map<string,double> interface.
//
// Expects the lwtnn *graph* JSON format (use converters/sequential2graph.py
// to convert from the sequential format).
//
// Usage:
//   lwtnn-bench-fastgraph-f32 <graph_config.json> [--n-inferences N]
//
#include "lwtnn/generic/FastGraph.hh"
#include "lwtnn/parse_json.hh"
#include "lwtnn/InputOrder.hh"
#include "test_utilities.hh"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
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
                  << " <graph_config.json> [--n-inferences N]\n";
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
    auto config = lwt::parse_json_graph(config_file);

    // ── build InputOrder: one scalar node, variables in config order ─────────
    lwt::InputOrder order;
    for (const auto& node : config.inputs) {
        std::vector<std::string> names;
        for (const auto& var : node.variables) names.push_back(var.name);
        order.scalar.emplace_back(node.name, names);
    }

    lwt::generic::FastGraph<float> net(config, order);

    // ── build test input: 1-D ramp in the same order as config.inputs ────────
    // ramp() returns the raw value such that after (raw+offset)*scale the
    // result is linspace(-1,1,n)[i].  For our MLP (offset=0, scale=1) the
    // raw value equals the normalised value.
    const auto& node0_vars = config.inputs.at(0).variables;
    const std::size_t n_inputs = node0_vars.size();
    lwt::VectorX<float> inv(n_inputs);
    for (std::size_t i = 0; i < n_inputs; ++i)
        inv(i) = static_cast<float>(ramp(node0_vars[i], i, n_inputs));

    std::vector<lwt::VectorX<float>> scalars = {inv};

    // ── warm-up ───────────────────────────────────────────────────────────────
    auto warm = net.compute(scalars);

    // ── timed loop ────────────────────────────────────────────────────────────
    using Clock = std::chrono::steady_clock;
    using us    = std::chrono::duration<double, std::micro>;

    std::vector<double> times_us;
    times_us.reserve(n_inferences);

    for (int i = 0; i < n_inferences; ++i) {
        auto t0  = Clock::now();
        auto out = net.compute(scalars);
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
    auto final_out = net.compute(scalars);

    // ── key=value output ──────────────────────────────────────────────────────
    std::cout << "backend=lwtnn-fastgraph-f32\n";
    std::cout << "n_inferences=" << n_inferences << "\n";
    std::printf("inference_mean_us=%.3f\n",   mean);
    std::printf("inference_min_us=%.3f\n",    min);
    std::printf("inference_median_us=%.3f\n", median);
    std::printf("inference_p99_us=%.3f\n",    p99);
    std::printf("inference_max_us=%.3f\n",    max);
    std::printf("peak_rss_kb=%ld\n",          rss);
    std::cout << "outputs=";
    for (int i = 0; i < final_out.size(); ++i) {
        if (i > 0) std::cout << " ";
        std::printf("out_%d=%.6f", i, static_cast<double>(final_out(i)));
    }
    std::cout << "\n";
    return 0;
}
