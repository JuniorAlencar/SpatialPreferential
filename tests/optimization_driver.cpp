#include "SamuraI.hpp"

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

// Shared by the old and new implementations, so regression measurements use
// the same compiler flags, inputs, public API and output precision.
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1]);
    const auto config = nlohmann::json::parse(input);
    samargs args{};
    args.num_vertices = config.at("n");
    args.m0 = config.at("m0");
    args.dim = config.at("dim");
    args.alpha_a = config.at("alpha_a");
    args.alpha_g = config.at("alpha_g");
    args.seed = config.at("seed");
    args.r_min = config.value("r_min", 1.0);
    args.r_max = config.value("r_max", 1e7);
    args.run_mode = 3;
    const int blocks = config.value("blocks", 100);
    const bool excess = config.value("excess", true);
    SamuraI graph(args);
    nlohmann::json result, times;
    auto timed = [&](const char* name, auto operation) {
        const auto start = std::chrono::steady_clock::now();
        operation();
        times[name] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    };
    timed("generation", [&] { graph.createGraph(); });
    timed("newman", [&] {
        const auto r = graph.computeAssortativityCoefficientNewmanDAGJK(blocks, excess);
        result["newman"] = r.R;
        result["newman_error"] = r.error;
    });
    timed("spearman", [&] {
        const auto r = graph.computeRankAssortativitySpearmanDAGJK(blocks, excess);
        result["spearman"] = r.R;
        result["spearman_error"] = r.error;
    });
    timed("clustering", [&] { result["clustering"] = graph.computeClusterCoefficient(); });
    timed("dijkstra", [&] {
        const auto paths = graph.computeGlobalNavigationDijkstraExact();
        result["weighted_mean"] = paths.shortestpath;
        result["weighted_diameter"] = paths.diamater;
        result["weighted_se"] = paths.shortestpath_se;
        result["coverage"] = paths.coverage;
    });
    timed("bfs", [&] {
        const auto paths = graph.computeGlobalNavigation_BFS();
        result["hop_mean"] = paths.shortestpath;
        result["hop_diameter"] = paths.diamater;
    });
    timed("gml", [&] { graph.writeGML(argv[2]); });
    result["seconds"] = times;
    std::cout << result.dump() << '\n';
}
