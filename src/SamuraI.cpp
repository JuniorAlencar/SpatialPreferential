#include "SamuraI.hpp"
#include "eigen_vec.hpp"
#include "randUtils.hpp"

#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/graph/breadth_first_search.hpp> //BFS
#include <boost/graph/visitors.hpp>
#include <boost/array.hpp>
#include<zlib.h>
#include <limits>

inline bool cmp(vertex_prop_double p1, vertex_prop_double p2) { return p1.second < p2.second; }

void SamuraI::CreateNetwork(){
    hrand rnd(m_seed);
    const int m0 = m_m0; // >= 1
    if (m0 < 1) throw std::runtime_error("m_m0 deve ser >= 1");
    if (m_num_vertices < m0 + 1)
        throw std::runtime_error("Para grau minimo m0, exija m_num_vertices >= m0 + 1");

    // Opcional: limpar estado, se necessário no seu contexto
    // G.clear(); pos.clear();
    // sum_positions.setZero(); center_of_mass.setZero();

    auto add_pos = [&](int i){
        double radius = rnd.inv_power_law_distributed(m_r_min, m_r_max, m_alpha_g); // ~ r^{-alpha_G}
        Vector4d dp   = rnd.uniform_hypersphere(m_dim);
        Vector4d p    = center_of_mass + radius * dp;
        pos.push_back(p);
        sum_positions += p;
        center_of_mass = sum_positions / double(i + 1);
    };

    // =========
    // 1) Núcleo inicial: clique K_{m0+1}
    //    -> cada vértice começa com grau exatamente m0.
    // =========
    const int m_init = m0 + 1;

    for (int i = 0; i < m_init; ++i){
        boost::add_vertex(G);
        add_pos(i);
    }
    // Conecta todos os pares (clique completo)
    for (int u = 0; u < m_init; ++u){
        for (int v = u + 1; v < m_init; ++v){
            // se seu tipo de grafo permitir multi-arestas, proteja:
            if (!boost::edge(u, v, G).second) boost::add_edge(u, v, G);
        }
    }
    // Até aqui: todos os vértices têm grau == m0

    // =========
    // 2) Crescimento: cada novo vértice faz m0 conexões distintas
    // =========
    for (int i = m_init; i < m_num_vertices; ++i){
        boost::add_vertex(G);
        add_pos(i);
        vertex_t New = i;

        // cada novo vértice nasce com exatamente m0 arestas
        const int links_to_add = m0;

        std::unordered_set<vertex_t> chosen;
        chosen.reserve(links_to_add);

        for (int ell = 0; ell < links_to_add; ++ell){
            // Recalcula pesos com graus atualizados e evitando alvos já escolhidos
            std::vector<vertex_prop_double> prob; // pair<vertex, weight>
            prob.reserve(static_cast<size_t>(i));
            const double exponent = -0.5 * m_alpha_a; // (r^2)^{-alpha_a/2} = r^{-alpha_a}
            double p_total = 0.0;

            for (int u = 0; u < i; ++u){
                vertex_t v = u;
                if (chosen.count(v)) continue; // sem duplicata de alvo

                int k_v = static_cast<int>(boost::degree(v, G));
                Vector4d Ruv = pos[v] - pos[New];
                double Ruv_SQ = Ruv.transpose() * Ruv;
                if (Ruv_SQ <= 0.0) Ruv_SQ = std::numeric_limits<double>::min(); // guarda numérico

                double w = static_cast<double>(k_v) * std::pow(Ruv_SQ, exponent);
                // se quiser um piso para evitar "sumir" alvos com k_v=0 (não ocorre aqui):
                // if (w <= 0) w = 0;
                p_total += w;
                prob.emplace_back(v, w);
            }

            // fallback robusto (patológico): se não houver pesos válidos, conecte aleatório distinto
            if (p_total <= 0.0 || prob.empty()){
                int trials = 0;
                while (trials++ < 1000){
                    int v = rnd.uniform_int(0, i - 1);
                    if (!chosen.count(v) && !boost::edge(v, New, G).second){
                        boost::add_edge(v, New, G);
                        chosen.insert(v);
                        break;
                    }
                }
                continue;
            }

            // amostragem por cumsum
            double r = rnd.uniform_real(0.0, 1.0);
            double cumsum = 0.0;
            for (auto &pr : prob){
                pr.second /= p_total; // normaliza
                cumsum += pr.second;
                if (cumsum > r){
                    vertex_t vstar = pr.first;
                    if (!boost::edge(vstar, New, G).second){
                        boost::add_edge(vstar, New, G);
                        chosen.insert(vstar);
                    } else {
                        // em grafos que não aceitam multi-arestas, tente outro (proteção extra)
                        // procura o primeiro não escolhido e sem aresta
                        for (auto &alt : prob){
                            if (!chosen.count(alt.first) && !boost::edge(alt.first, New, G).second){
                                boost::add_edge(alt.first, New, G);
                                chosen.insert(alt.first);
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
        // Agora: deg(New) == m0; vértices antigos mantêm deg >= m0 (nunca diminuem).
    }

//    (Opcional) verificação final em debug:
    for (int v = 0; v < m_num_vertices; ++v){
        using degree_size_t = boost::graph_traits<decltype(G)>::degree_size_type;

        degree_size_t deg = boost::degree(v, G);
        if (deg < static_cast<degree_size_t>(m0)) {
            throw std::runtime_error("Grau < m0 detectado (inconsistência na construção).");
        }
    }
}

Navigation SamuraI::computeGlobalNavigation(){
	struct Navigation BFS;

    double meanShortestPath = 0;
    //int count = 0;
    std::vector<int> d(m_num_vertices, 0);  // vector for diamater
    int aux = 0;                           // auxiliary value for the diameter
    
    for(auto u : boost::make_iterator_range(vertices(G))){
        //boost::array<int, 100000> distances{{0}};
        std::vector<int> distances(m_num_vertices,0);  // vector for diamater
        breadth_first_search(G, u, visitor(make_bfs_visitor(record_distances(&distances[0], on_tree_edge()))));
        // breadth_first_search(G, u, visitor(
        //      make_bfs_visitor( record_distances(distances.begin(), 
        //      on_tree_edge{}))));
    
        for (auto e=distances.begin(); e != distances.end(); ++e){
             meanShortestPath += *e;
             //++ count;
             if(*e > d[aux])
                d[aux] = *e;
      }
    }
    int dia = *max_element(d.begin(), d.end());
    double count = num_vertices(G)*(num_vertices(G)-1);
    meanShortestPath /= count;
    
    BFS.shortestpath_BFS = meanShortestPath;
    BFS.diamater_BFS = dia;
    
    return BFS;
}

double SamuraI::computeClusterCoefficient() {
    // Container com coeficientes locais
    ClusteringContainer c_container(num_vertices(G));

    // Criação correta do property_map usando make_iterator_property_map
    ClusteringMap c_map = make_iterator_property_map(c_container.begin(), get(vertex_index, G));

    // Calcula e retorna o coeficiente global
    double global_clustering = all_clustering_coefficients(G, c_map);

    return global_clustering;
}


double SamuraI::computeAssortativityCoefficientNewman() {
    const double M = static_cast<double>(boost::num_edges(G)); // número de arestas reais

    double T1 = 0.0, T2 = 0.0, T3 = 0.0;

    for (auto e : boost::make_iterator_range(boost::edges(G))) {
        int u_deg = boost::degree(boost::source(e, G), G);
        int v_deg = boost::degree(boost::target(e, G), G);

        T1 += static_cast<double>(u_deg) * static_cast<double>(v_deg);
        T2 += 0.5 * (static_cast<double>(u_deg) + static_cast<double>(v_deg));
        T3 += 0.5 * (std::pow(u_deg, 2) + std::pow(v_deg, 2));
    }

    T1 /= M;
    T2 /= M;
    T3 /= M;

    double denom = T3 - T2 * T2;
    if (std::abs(denom) < 1e-12)
        return 0.0; // rede degenerada (ex.: todos graus iguais)

    double R = (T1 - T2 * T2) / denom;
    return R;
}


static std::vector<double> compute_ranks_average_ties(const std::vector<double>& vals) {
    int n = (int)vals.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    // ordenar índices por valor correspondente
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return vals[a] < vals[b];
    });

    std::vector<double> ranks(n);
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && vals[idx[j]] == vals[idx[i]]) ++j;
        // valores iguais do i..j-1 -> atribuir rank médio (1-based)
        double rank_sum = 0.0;
        for (int t = i; t < j; ++t) rank_sum += (t + 1); // ranks 1..n
        double avg_rank = rank_sum / double(j - i);
        for (int t = i; t < j; ++t) ranks[idx[t]] = avg_rank;
        i = j;
    }
    return ranks;
}

// Calcula correlação Pearson de dois vetores (mesmo tamanho). Retorna NaN se inválido.
static double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    int n = (int)x.size();
    if (n == 0 || (int)y.size() != n) return std::numeric_limits<double>::quiet_NaN();
    double mean_x = 0.0, mean_y = 0.0;
    for (int i = 0; i < n; ++i) { mean_x += x[i]; mean_y += y[i]; }
    mean_x /= n; mean_y /= n;
    double num = 0.0, sx = 0.0, sy = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        num += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    double denom = std::sqrt(sx * sy);
    if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return num / denom;
}

// Método a adicionar na sua classe (ajuste o tipo de G conforme seu grafo)
double SamuraI::computeRankAssortativitySpearman() {
        
    using boost::make_iterator_range;

    // colecione pares de graus (um par por aresta)
    std::vector<double> deg_u_list;
    std::vector<double> deg_v_list;
    deg_u_list.reserve(boost::num_edges(G));
    deg_v_list.reserve(boost::num_edges(G));

    for (auto e : make_iterator_range(boost::edges(G))) {
        auto u = boost::source(e, G);
        auto v = boost::target(e, G);
        if (u == v) continue; // ignorar self-loops, se existirem
        double ku = static_cast<double>(boost::degree(u, G));
        double kv = static_cast<double>(boost::degree(v, G));
        deg_u_list.push_back(ku);
        deg_v_list.push_back(kv);
    }

    int M = (int)deg_u_list.size();
    if (M == 0) return 0.0;

    // criamos vetores de todos os valores que serão ranqueados separadamente:
    // Para Spearman clássico, os ranks devem ser obtidos sobre cada conjunto separadamente?
    // Não — ranqueamos *globalmente* cada componente separadamente. Isto é:
    // - ranks para deg_u_list (cada entrada corresponde a uma extremidade "u" da aresta)
    // - ranks para deg_v_list (cada entrada corresponde a uma extremidade "v")
    //
    // Uma alternativa comum é ranquear os graus de TODOS os nós (não por aresta). 
    // Aqui faremos a versão por aresta (rankeando os vetores deg_u_list e deg_v_list) 
    // que é consistente com cálculo de Spearman entre as extremidades.
    std::vector<double> ranks_u = compute_ranks_average_ties(deg_u_list);
    std::vector<double> ranks_v = compute_ranks_average_ties(deg_v_list);

    // agora a correlação de Pearson entre os ranks é o Spearman rho
    double rho = pearson_correlation(ranks_u, ranks_v);

    // proteção numérica / limites
    if (std::isnan(rho)) return 0.0;
    if (rho > 1.0 && rho < 1.0 + 1e-12) rho = 1.0;
    if (rho < -1.0 && rho > -1.0 - 1e-12) rho = -1.0;
    
    return rho;
}

Navigation SamuraI::computeGlobalNavigation_Astar() {
    Navigation AStar;
    double meanShortestPath = 0.0;
    std::vector<double> d(m_num_vertices, 0.0); // vetor para calcular o diâmetro
    int aux = 0;

    // Mapa de pesos das arestas com base na distância entre os sítios
    std::map<edge_t, double> weight_map_data;
    for (auto e : boost::make_iterator_range(edges(G))) {
        vertex_t u = source(e, G);
        vertex_t v = target(e, G);
        weight_map_data[e] = (pos[u] - pos[v]).norm(); // distância euclidiana
    }
    auto edge_weight = boost::make_assoc_property_map(weight_map_data);

    for (vertex_t u : boost::make_iterator_range(vertices(G))) {
        // Mapas de distância e predecessores com segurança total via Boost
        boost::vector_property_map<double> distances(num_vertices(G));
        boost::vector_property_map<vertex_t> predecessors(num_vertices(G));

        // Heurística nula (Dijkstra), pois estamos fazendo origem única → todos
        auto heuristic = [](vertex_t) { return 0.0; };

        boost::astar_search(
            G,
            u,
            heuristic,
            boost::weight_map(edge_weight)
                .distance_map(distances)
                .predecessor_map(predecessors)
                .visitor(boost::make_astar_visitor(boost::null_visitor()))
        );

        for (vertex_t v : boost::make_iterator_range(vertices(G))) {
            if (v != u && distances[v] < std::numeric_limits<double>::max()) {
                meanShortestPath += distances[v];
                if (distances[v] > d[aux])
                    d[aux] = distances[v];
            }
        }
    }

    double count = static_cast<double>(num_vertices(G)) * (num_vertices(G) - 1);
    meanShortestPath /= count;

    AStar.shortestpath_WEIGHT = meanShortestPath;
    AStar.diamater_WEIGHT = static_cast<int>(*std::max_element(d.begin(), d.end()));
    return AStar;
}


void SamuraI::writeDegrees(std::string fname){
    std::cout << fname << std::endl;

    gzFile fi = gzopen(fname.c_str(),"wb");
    gzprintf(fi,"k,\r\n");
    for(int i=0;i<m_num_vertices;i++){
        vertex_t v = i;
        gzprintf(fi,"%d,\r\n",boost::degree(v,G));
        }
    gzclose(fi);
}

void SamuraI::writeConnections(std::string fname){
    std::cout << fname << std::endl;

    gzFile fi = gzopen(fname.c_str(),"wb");
    gzprintf(fi,"#Node1,");
    gzprintf(fi,"#Node2,\r\n");
    for (auto e : boost::make_iterator_range(boost::edges(G))) {
        gzprintf(fi,"%d,",boost::source(e, G));
        gzprintf(fi,"%d\r\n",boost::target(e, G));
    }
    gzclose(fi);
}

void SamuraI::writeGML(std::string fname){
    std::cout << fname << std::endl;
    
    gzFile fi = gzopen(fname.c_str(),"wb");

    gzprintf(fi, "graph\n");
    gzprintf(fi, "[\n");
    gzprintf(fi, "  Creator \"Gephi\"\n");
    gzprintf(fi, "  undirected 0\n");
    
//    for (auto v : boost::make_iterator_range(boost::vertices(G))) {
      for (int i=0;i<m_num_vertices;i++) {
	vertex_t v = i;
        gzprintf(fi, "node\n");
        gzprintf(fi, "[\n");
        gzprintf(fi, "id %d\n", i);
        gzprintf(fi, "label %d\n",i);
        gzprintf(fi, "graphics\n");
        gzprintf(fi, "[\n");
        gzprintf(fi, "x %f\n", pos[v](0));
        gzprintf(fi, "y %f\n", pos[v](1));
        gzprintf(fi, "z %f\n", pos[v](2));
        gzprintf(fi,"]\n");
        gzprintf(fi, "degree %d\n", boost::degree(v,G));
        gzprintf(fi,"]\n");
    }
    
     for (auto e : boost::make_iterator_range(boost::edges(G))){
         Vector4d Ruv = pos[boost::source(e, G)] - pos[boost::target(e, G)];
         gzprintf(fi,"edge\n");
         gzprintf(fi,"[\n");
         gzprintf(fi, "source %d\n", boost::source(e, G));
         gzprintf(fi, "target %d\n", boost::target(e, G));
         gzprintf(fi, "distance %.15f\n", pow(Ruv.transpose() * Ruv, 0.5));
         gzprintf(fi, "]\n");
     }
     gzprintf(fi,"]");
     gzclose(fi);
 }

void SamuraI::clear() {
    G.clear();
    pos.clear();
};

