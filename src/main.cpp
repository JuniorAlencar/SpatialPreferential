#include "SamuraI.hpp"
#include "SamuraIConfig.hpp"
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <boost/graph/random.hpp>
#include <random>

#include <chrono>
#include <cstdlib>
#include <iomanip>  // for setprecision
#include <sstream>
#include <unistd.h>

// comentarion
using json = nlohmann::json;

namespace fs = boost::filesystem;

using namespace std;

enum RunMode : int {
    PROPS_ONLY = 1,
    NETWORK_ONLY = 2,
    BOTH = 3
};


class ProgressReporter {
    using Clock = std::chrono::steady_clock;
    json benchmark = json::object();
    std::string current_stage;
    Clock::time_point started;
    std::string metrics_path;
    int seed;

    static std::string duration(double seconds) {
        std::ostringstream out;
        if (seconds >= 3600.0) {
            out << static_cast<long long>(seconds / 3600) << "h "
                << static_cast<int>(seconds / 60) % 60 << "min";
        } else if (seconds >= 60.0) {
            out << static_cast<int>(seconds / 60) << "min "
                << static_cast<int>(seconds) % 60 << "s";
        } else {
            out << std::fixed << std::setprecision(1) << seconds << "s";
        }
        return out.str();
    }

    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - started).count();
    }

    double reference() const {
        auto it = benchmark.find(current_stage);
        return it != benchmark.end() && it->is_number() ? it->get<double>() : 0.0;
    }

    void print(const std::string& message) const {
        std::cerr << "[seed=" << seed << " pid=" << getpid() << "] "
                  << current_stage << ": " << message << '\n';
    }

public:
    explicit ProgressReporter(int current_seed) : seed(current_seed) {
        if (const char* value = std::getenv("SPATIAL_STAGE_BENCHMARK")) {
            auto parsed = json::parse(value, nullptr, false);
            if (parsed.is_object()) benchmark = std::move(parsed);
        }
        if (const char* value = std::getenv("SPATIAL_STAGE_METRICS_FILE"))
            metrics_path = value;
    }

    void begin(const std::string& stage) {
        current_stage = stage;
        started = Clock::now();
        if (reference() > 0.0)
            print("iniciado | tempo estimado pelo benchmark " + duration(reference()));
        else
            print("iniciado | medindo tempo desta etapa");
    }

    void update(std::size_t completed, std::size_t total) const {
        if (total == 0) return;
        const double fraction = static_cast<double>(completed) / total;
        const double live_total = elapsed() / fraction;
        const double benchmark_total = reference();
        const double estimated_total = benchmark_total > 0.0
            ? std::max(benchmark_total, live_total) : live_total;
        print(std::to_string(completed) + " / " + std::to_string(total)
              + " | restante ~" + duration(std::max(0.0, estimated_total * (1.0 - fraction)))
              + (benchmark_total > 0.0 ? " (benchmark + ritmo atual)" : " (ritmo atual)"));
    }

    void end() const {
        const double seconds = elapsed();
        print("concluído em " + duration(seconds));
        if (!metrics_path.empty()) {
            std::ofstream out(metrics_path, std::ios::app);
            out << json{{"stage", current_stage}, {"seconds", seconds}}.dump() << '\n';
        }
    }
};

samargs read_parametes(const string& fname) {
    std::ifstream f(fname.c_str());
    json data = json::parse(f);

    samargs xargs;
    xargs.num_vertices = data.value("num_vertices", 128);
    xargs.alpha_a = data.value("alpha_a", 1.0);
    xargs.alpha_g = data.value("alpha_g", 1.0);
    xargs.r_min = data.value("r_min", 1.0);
    xargs.r_max = data.value("r_max", 1e7);
    xargs.dim = data.value("dim", 3);
    xargs.seed = data.value("seed", 1234);
    xargs.m0 = data.value("m0", 1);
    xargs.run_mode     = data.value("run_mode", static_cast<int>(NETWORK_ONLY));
    
    if (xargs.seed < 0) {
        // std::time_t now = std::time(0);
        // xargs.seed = now % 10000;
        std::random_device rd;
        boost::mt19937 Gen(rd());
        boost::random::uniform_int_distribution<int> dist(1, 2147483647);
        int now = dist(Gen);
        xargs.seed = now;
    }

    // leitura robusta para props


    return xargs;
}

int main(int argc, char* argv[]) {

    // Path to exec
    fs::path exe_path = fs::canonical("/proc/self/exe");

    // Build to workscape/
    fs::path workspace = exe_path.parent_path().parent_path();

    // Folder in same lavel of src/ and build/
    fs::path data_dir = workspace / "data";

    // Create folder
    fs::create_directories(data_dir);

    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " file.json" << endl;
        return 1;
    }

    // read parameters from json file
    samargs xargs = read_parametes(argv[1]);
    std::ifstream options_file(argv[1]);
    const json options = json::parse(options_file);
    const bool compute_dijkstra = options.value("compute_dijkstra", false);
    const bool compute_spearman = options.value("compute_spearman", false);

    // set filenames
    const string data_path = data_dir.string();
    const string n_folder = data_path + "/N_" + to_string(xargs.num_vertices) +
                            "/m0_" + to_string(xargs.m0);

    ostringstream alpha_path;
    alpha_path << n_folder << "/dim_" << xargs.dim
               << "/alpha_a_" << fixed << setprecision(2) << xargs.alpha_a
               << "_alpha_g_" << fixed << setprecision(2) << xargs.alpha_g;
    const string alpha_folder = alpha_path.str();
    const string gml_folder = alpha_folder + "/gml";
    const string prop_folder = alpha_folder + "/prop";
    const string L_coast_file = prop_folder + "/L_coast_" + to_string(xargs.seed) + ".csv";
    const string prop_file = prop_folder + "/prop_" + to_string(xargs.seed) + ".csv";
    const string gml_file = gml_folder + "/gml_" + to_string(xargs.seed) + ".gml.gz";
    const string time_process_file = alpha_folder + "/time_process_seconds.txt";
    
    // create dir
    fs::create_directories(alpha_folder);
    fs::create_directories(gml_folder);
    fs::create_directories(prop_folder);

    SamuraI S(xargs);
    ProgressReporter progress(xargs.seed);
    S.setProgressCallback([&](std::size_t completed, std::size_t total) {
        progress.update(completed, total);
    });
    auto run_stage = [&](const std::string& name, auto&& operation) {
        progress.begin(name);
        operation();
        progress.end();
    };
    run_stage("create_network", [&] { S.createGraph(); });
    
    auto run_mode = static_cast<RunMode>(xargs.run_mode);
    
    if (run_mode == NETWORK_ONLY || run_mode == BOTH)
        run_stage("save_data/GML", [&] { S.writeGML(gml_file); });

    if (run_mode == PROPS_ONLY || run_mode == BOTH) {
        R_ass_Newman r_newman{};
        R_ass_Spearman r_spearman{};
        double C = 0.0;
        Navigation_COST shortest_cost;
        Navigation_BFS shortest_bfs;
        run_stage("compute_properties/Newman", [&] {
            r_newman = S.computeAssortativityCoefficientNewmanDAGJK(100, true);
        });
        if (compute_spearman)
            run_stage("compute_properties/Spearman", [&] {
                r_spearman = S.computeRankAssortativitySpearmanDAGJK(100, true);
            });
        run_stage("compute_properties/Clustering", [&] {
            C = S.computeClusterCoefficient();
        });
        if (compute_dijkstra)
            run_stage("compute_properties/Shortest_Dijkstra", [&] {
                shortest_cost = S.computeGlobalNavigationDijkstraExact();
            });
        run_stage("compute_properties/Short_BFS", [&] {
            shortest_bfs = S.computeGlobalNavigation_BFS();
        });
        run_stage("save_data/properties", [&] {
            std::cout << prop_file << std::endl;
            std::ofstream pout(prop_file);
            pout << "#MeanShortestPathDijstrika, " << "#MeanShortestPathBFS, "
                << "#Ass_Spearman, " << "#Ass_Spearman_error, "
                << "#Ass_Newman, " << "#Ass_Newman_error, "
                << "#ClusterCoefficient\n";
            if (compute_dijkstra) pout << shortest_cost.shortestpath;
            pout << "," << shortest_bfs.shortestpath << ",";
            if (compute_spearman) pout << r_spearman.R << "," << r_spearman.error;
            else pout << ",";
            pout << "," << r_newman.R << "," << r_newman.error << "," << C << "\n";
        });
    } else if (run_mode != NETWORK_ONLY) {
        std::cerr << "[warn] run_mode inválido (" << xargs.run_mode
                  << "). Use 1 (props), 2 (network), 3 (both).\n";
    }

    S.clear();
    
    cout << time_process_file << endl;
    // Gen file to calculate time to run process
    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration in seconds with 5 decimal places
    auto duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    // Save the duration to a file
    std::ofstream file(time_process_file, std::ios::app);  // Open file in append mode
    if (file.is_open()) {
        // Set precision to 5 decimal places
        
        file << std::fixed << std::setprecision(5) << duration_seconds << std::endl;
        file.close();
        std::cout << "Execution time saved to file." << std::endl;
    } else {
        std::cerr << "Error opening file for writing." << std::endl;
    }
    // -----------------
    return 0;
}
