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
#include <iomanip>  // for setprecision

// comentarion
using json = nlohmann::json;

namespace fs = boost::filesystem;

using namespace std;

enum RunMode : int {
    PROPS_ONLY = 1,
    NETWORK_ONLY = 2,
    BOTH = 3
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
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " file.json" << endl;
        return 1;
    }

    // read parameters from json file
    samargs xargs = read_parametes(argv[1]);

    // set filenames
    char n_folder[50], alpha_folder[160], gml_folder[170], prop_folder[170], prop_file[200], gml_file[200], L_coast_file[200] ,time_process_file[200];
    sprintf(n_folder, "./N_%d/m0_%d", xargs.num_vertices, xargs.m0);
    sprintf(alpha_folder, "%s/dim_%d/alpha_a_%1.2f_alpha_g_%1.2f", n_folder, xargs.dim, xargs.alpha_a, xargs.alpha_g);
    sprintf(gml_folder, "%s/gml", alpha_folder);
    sprintf(prop_folder, "%s/prop", alpha_folder);
    sprintf(L_coast_file, "%s/L_coast_%d.csv", prop_folder, xargs.seed);
    sprintf(prop_file, "%s/prop_%d.csv", prop_folder, xargs.seed);
    sprintf(gml_file, "%s/gml_%d.gml.gz", gml_folder, xargs.seed);
    sprintf(time_process_file, "%s/time_process_seconds.txt", alpha_folder);
    
    // create dir
    fs::create_directories(alpha_folder);
    fs::create_directories(gml_folder);
    fs::create_directories(prop_folder);

    SamuraI S(xargs);
    S.createGraph();
    
    auto run_mode = static_cast<RunMode>(xargs.run_mode);
    
    switch (run_mode) {
    case PROPS_ONLY: {
        R_ass_Newman   r_newman   = S.computeAssortativityCoefficientNewmanDAGJK(/*B=*/100, /*use_excess=*/true);
        R_ass_Spearman r_spearman = S.computeRankAssortativitySpearmanDAGJK(/*B=*/100, /*use_excess=*/true);
        double C = S.computeClusterCoefficient();

        Navigation_COST AStar = S.computeGlobalNavigationDijkstraExact();
        Navigation_BFS  bfs   = S.computeGlobalNavigation_BFS();

        std::cout << prop_file << std::endl;
        std::ofstream pout(prop_file);
        pout << "#MeanShortestPathDijstrika, " << "#MeanShortestPathBFS, "
            << "#Ass_Spearman, " << "#Ass_Spearman_error, "
            << "#Ass_Newman, "  << "#Ass_Newman_error, "
            << "#ClusterCoefficient\n";
        pout << AStar.shortestpath << "," << bfs.shortestpath << ","
            << r_spearman.R << "," << r_spearman.error  << ","
            << r_newman.R   << "," << r_newman.error    << ","
            << C << "\n";
        break;
    }
    case NETWORK_ONLY: {
        S.writeGML(gml_file);
        break;
    }
    case BOTH: {
    // 1) salva rede
    S.writeGML(gml_file);

    // 2) calcula e salva propriedades (reuso do bloco acima)
    R_ass_Newman   r_newman   = S.computeAssortativityCoefficientNewmanDAGJK(/*B=*/100, /*use_excess=*/true);
    R_ass_Spearman r_spearman = S.computeRankAssortativitySpearmanDAGJK(/*B=*/100, /*use_excess=*/true);
    double C = S.computeClusterCoefficient();

    Navigation_COST AStar = S.computeGlobalNavigationDijkstraExact();
    Navigation_BFS  bfs   = S.computeGlobalNavigation_BFS();

    std::cout << prop_file << std::endl;
    std::ofstream pout(prop_file);
    pout << "#MeanShortestPathDijstrika, " << "#MeanShortestPathBFS, "
         << "#Ass_Spearman, " << "#Ass_Spearman_error, "
         << "#Ass_Newman, "  << "#Ass_Newman_error, "
         << "#ClusterCoefficient\n";
    pout << AStar.shortestpath << "," << bfs.shortestpath << ","
         << r_spearman.R << "," << r_spearman.error  << ","
         << r_newman.R   << "," << r_newman.error    << ","
         << C << "\n";
    break;
    }
    default:
        std::cerr << "[warn] run_mode inválido (" << xargs.run_mode
                << "). Use 1 (props), 2 (network), 3 (both).\n";
        break;
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
