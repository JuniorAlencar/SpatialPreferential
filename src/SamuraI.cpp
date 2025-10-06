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
    // (re)inicia o map de pesos com functors que conhecem G
    edge_weight_store = decltype(edge_weight_store)(0, EdgeHash{&G}, EdgeEq{&G});
    edge_weight_store.reserve(static_cast<std::size_t>(1.5 * m_num_vertices * m_m0));

    hrand rnd(m_seed);
    const int m0 = m_m0;
    if (m0 < 1) throw std::runtime_error("m_m0 deve ser >= 1");
    if (m_num_vertices < m0 + 1)
        throw std::runtime_error("Para grau minimo m0, exija m_num_vertices >= m0 + 1");

    auto add_pos = [&](int i){
        double radius = rnd.inv_power_law_distributed(m_r_min, m_r_max, m_alpha_g); // ~ r^{-alpha_G}
        Vector4d dp   = rnd.uniform_hypersphere(m_dim);
        Vector4d p    = center_of_mass + radius * dp;
        pos.push_back(p);
        sum_positions += p;
        center_of_mass = sum_positions / double(i + 1);
    };

    // 1) Núcleo inicial
    const int m_init = m0 + 1;

    for (int i = 0; i < m_init; ++i){
        boost::add_vertex(G);
        add_pos(i);
    }
    // clique completo com peso
    for (int u = 0; u < m_init; ++u){
        for (int v = u + 1; v < m_init; ++v){
            if (!boost::edge(u, v, G).second) add_weighted_edge(u, v);
        }
    }

    // 2) Crescimento
    for (int i = m_init; i < m_num_vertices; ++i){
        boost::add_vertex(G);
        add_pos(i);
        vertex_t New = i;

        const int links_to_add = m0;
        std::unordered_set<vertex_t> chosen;
        chosen.reserve(links_to_add);

        for (int ell = 0; ell < links_to_add; ++ell){
            std::vector<vertex_prop_double> prob; // pair<vertex, weight>
            prob.reserve(static_cast<size_t>(i));
            const double exponent = -0.5 * m_alpha_a; // (r^2)^{-alpha_a/2}
            double p_total = 0.0;

            for (int u = 0; u < i; ++u){
                vertex_t v = u;
                if (chosen.count(v)) continue;

                int k_v = static_cast<int>(boost::degree(v, G));
                Vector4d Ruv = pos[v] - pos[New];
                double Ruv_SQ = Ruv.transpose() * Ruv;
                if (Ruv_SQ <= 0.0) Ruv_SQ = std::numeric_limits<double>::min();

                double w = static_cast<double>(k_v) * std::pow(Ruv_SQ, exponent);
                p_total += w;
                prob.emplace_back(v, w);
            }

            // fallback: aleatório válido
            if (p_total <= 0.0 || prob.empty()){
                int trials = 0;
                while (trials++ < 1000){
                    int v = rnd.uniform_int(0, i - 1);
                    if (!chosen.count(v) && !boost::edge(v, New, G).second){
                        add_weighted_edge(v, New);
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
                pr.second /= p_total;
                cumsum += pr.second;
                if (cumsum > r){
                    vertex_t vstar = pr.first;
                    if (!boost::edge(vstar, New, G).second){
                        add_weighted_edge(vstar, New);
                        chosen.insert(vstar);
                    } else {
                        for (auto &alt : prob){
                            if (!chosen.count(alt.first) && !boost::edge(alt.first, New, G).second){
                                add_weighted_edge(alt.first, New);
                                chosen.insert(alt.first);
                                break;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // verificação final (opcional)
    for (int v = 0; v < m_num_vertices; ++v){
        using degree_size_t = boost::graph_traits<decltype(G)>::degree_size_type;
        degree_size_t deg = boost::degree(v, G);
        if (deg < static_cast<degree_size_t>(m0)) {
            throw std::runtime_error("Grau < m0 detectado (inconsistência na construção).");
        }
    }
}


Navigation_BFS SamuraI::computeGlobalNavigation_BFS(){
	struct Navigation_BFS BFS;

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
    
    BFS.shortestpath = meanShortestPath;
    BFS.diamater = dia;
    
    return BFS;
}


void SamuraI::add_weighted_edge(int u, int v) {
    auto [e, ok] = boost::add_edge(u, v, G);
    if (!ok) return;
    const Vector4d d  = pos[u] - pos[v];
    const double w2   = (d.transpose() * d);
    const double w    = std::sqrt(std::max(0.0, w2)); // distância física
    edge_weight_store[e] = w; // guarda no map externo
}


static inline int clamp_int(int x, int lo, int hi){ return std::max(lo, std::min(hi, x)); }

Navigation_COST SamuraI::computeGlobalNavigationDijkstraAuto(void) {
    using boost::make_iterator_range;
    using Graph  = decltype(G);
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using Edge   = typename boost::graph_traits<Graph>::edge_descriptor;

    Navigation_COST out;
    const std::size_t N = boost::num_vertices(G);
    if (N <= 1) { out.shortestpath = 0.0; out.diamater = 0.0; out.coverage = 1.0; return out; }

    auto vindex = get(boost::vertex_index, G);

    // ----------------- AUTOTUNE -----------------
    auto clamp_int = [](int x, int lo, int hi){ return std::max(lo, std::min(hi, x)); };
    int S = (int)std::ceil(2.0 * std::sqrt((double)N));
    S = clamp_int(S, 64, 512);
    if ((std::size_t)S > N) S = (int)N;

    int k_sweeps = (int)std::ceil(std::log2((double)N));
    k_sweeps = clamp_int(k_sweeps, 8, 32);

    // fontes (reprodutível com o mesmo seed)
    std::vector<Vertex> verts; verts.reserve(N);
    for (auto v : make_iterator_range(boost::vertices(G))) verts.push_back(v);
    std::mt19937_64 rng(static_cast<uint64_t>(m_seed) ^ 0x9e3779b97f4a7c15ULL);
    std::shuffle(verts.begin(), verts.end(), rng);
    verts.resize(S);

    // ---------- weight map por função com fallback ----------
    auto wmap = boost::make_function_property_map<Edge, double>(
        [&](const Edge& e) -> double {
            auto it = edge_weight_store.find(e);
            if (it != edge_weight_store.end()) return it->second;
            // fallback: calcula distância euclidiana
            auto u = boost::source(e, G);
            auto v = boost::target(e, G);
            const Vector4d d = pos[u] - pos[v];
            const double w2  = (d.transpose() * d);
            return std::sqrt(std::max(0.0, w2));
        }
    );

    // ---------- sanity: média do peso das arestas ----------
#ifndef NDEBUG
    {
        double sumw = 0.0; std::size_t cnt = 0;
        for (auto e : make_iterator_range(boost::edges(G))) {
            double w = wmap[e];
            sumw += w; ++cnt;
        }
        std::cerr << "[DBG] mean_edge_weight = " << (cnt? sumw/cnt : 0.0) << "\n";
    }
#endif

    // buffers de distâncias (weighted + hops)
    std::vector<double> dist(N, std::numeric_limits<double>::infinity());
    auto dist_map = boost::make_iterator_property_map(dist.begin(), vindex);
    std::vector<int> hop(N, INT_MAX);
    auto hop_map = boost::make_iterator_property_map(hop.begin(), vindex);

    // ---- médias (weighted e hops), mais cobertura ----
    double sum_all_w = 0.0;  std::uint64_t cnt_all_w = 0;
    double sum_all_h = 0.0;  std::uint64_t cnt_all_h = 0;
    std::vector<double> per_source_mean_w; per_source_mean_w.reserve(S);

    for (int i = 0; i < S; ++i) {
        Vertex s = verts[i];

        // weighted (Dijkstra)
        std::fill(dist.begin(), dist.end(), std::numeric_limits<double>::infinity());
        boost::dijkstra_shortest_paths(G, s, boost::weight_map(wmap).distance_map(dist_map));
        double sum_s_w = 0.0; std::uint64_t cnt_s_w = 0;

        // hops (BFS “implícito”: weights=1) – fazemos um Dijkstra com peso 1
        std::fill(hop.begin(), hop.end(), INT_MAX);
        // podemos simular peso 1 usando uma função constante
        auto unit_wmap = boost::make_function_property_map<Edge, double>([](const Edge&){ return 1.0; });
        // dist reutilizada? melhor não: usamos hop_map separado:
        // rodamos dijkstra com pesos 1 para obter hops (em grafos sem pesos negativos, equivale a BFS)
        std::vector<double> hop_as_double(N, std::numeric_limits<double>::infinity());
        auto hop_as_double_map = boost::make_iterator_property_map(hop_as_double.begin(), vindex);
        boost::dijkstra_shortest_paths(G, s, boost::weight_map(unit_wmap).distance_map(hop_as_double_map));
        double sum_s_h = 0.0; std::uint64_t cnt_s_h = 0;

        for (auto v : make_iterator_range(boost::vertices(G))) {
            if (v == s) continue;
            // weighted
            const double dw = dist[v];
            if (std::isfinite(dw)) { sum_s_w += dw; ++cnt_s_w; }
            // hops
            const double dh = hop_as_double[v];
            if (std::isfinite(dh)) { sum_s_h += dh; ++cnt_s_h; }
        }

        if (cnt_s_w > 0) { sum_all_w += sum_s_w; cnt_all_w += cnt_s_w; per_source_mean_w.push_back(sum_s_w/(double)cnt_s_w); }
        if (cnt_s_h > 0) { sum_all_h += sum_s_h; cnt_all_h += cnt_s_h; }
    }

    out.shortestpath = (cnt_all_w ? sum_all_w / (double)cnt_all_w : 0.0);
    out.coverage     = (N > 1) ? ( (double)cnt_all_w / (double)(S * (N - 1)) ) : 1.0;

    // erro (jackknife por fonte) – weighted
    if (per_source_mean_w.size() >= 2) {
        double mbar = 0.0; for (double m : per_source_mean_w) mbar += m; mbar /= (double)per_source_mean_w.size();
        double var = 0.0;  for (double m : per_source_mean_w) { double d = m - mbar; var += d*d; }
        var *= (double)(per_source_mean_w.size() - 1) / (double)per_source_mean_w.size();
        out.shortestpath_se = std::sqrt(std::max(0.0, var / (double)per_source_mean_w.size()));
    } else {
        out.shortestpath_se = 0.0;
    }

    // ---- diâmetro aproximado (k double-sweeps) com pesos corretos ----
    auto farthest = [&](Vertex src, double &maxd, Vertex &argmax){
        std::fill(dist.begin(), dist.end(), std::numeric_limits<double>::infinity());
        boost::dijkstra_shortest_paths(G, src, boost::weight_map(wmap).distance_map(dist_map));
        maxd = 0.0; argmax = src;
        for (auto v : make_iterator_range(boost::vertices(G))) {
            double dv = dist[v];
            if (std::isfinite(dv) && dv > maxd) { maxd = dv; argmax = v; }
        }
    };

    double best = 0.0;
    for (int t = 0; t < k_sweeps; ++t) {
        Vertex a = verts[t % S]; double d1; Vertex b;
        farthest(a, d1, b);
        double d2; Vertex c;
        farthest(b, d2, c);
        if (d1 > best) best = d1;
        if (d2 > best) best = d2;
    }
    out.diamater = best;

#ifndef NDEBUG
    // logs de diagnóstico
    double Lw = out.shortestpath;
    double Lh = (cnt_all_h ? (sum_all_h / (double)cnt_all_h) : 0.0);
    std::cerr << "[DBG] N=" << N
              << "  S=" << S
              << "  k_sweeps=" << k_sweeps
              << "  coverage=" << out.coverage
              << "  L_weighted=" << Lw
              << "  L_hops=" << Lh
              << "  ratio(Lw/Lh)=" << (Lh>0? Lw/Lh : -1.0)
              << "\n";
#endif

    return out;
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


// hashzinho estável para distribuir vértices em B buckets
static inline std::size_t mix_hash(std::size_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

/**
 * Assortatividade por grau (Newman 2002) com erro via DAGJK por blocos de vértices.
 * r = (T1 - T2^2) / (T3 - T2^2), com T1,T2,T3 médias por aresta.
 * @param B           número de blocos (ex.: 100)
 * @param use_excess  se true, usa (k-1) em vez de k
 */
R_ass_Newman SamuraI::computeAssortativityCoefficientNewmanDAGJK(int B, bool use_excess /*=false*/) {
    using boost::make_iterator_range;

    R_ass_Newman out;

    const std::size_t Mz = boost::num_edges(G);
    const std::size_t Nz = boost::num_vertices(G);
    if (Mz == 0 || Nz == 0) return out;
    if (B < 2) B = 2;

    auto vidx = get(boost::vertex_index, G);

    // ======================= r no grafo completo =======================
    double S1 = 0.0, S2 = 0.0, S3 = 0.0; // somatórios (não normalizados)
    std::size_t M_use = 0;

    for (auto e : make_iterator_range(boost::edges(G))) {
        auto u = boost::source(e, G);
        auto v = boost::target(e, G);
        if (u == v) continue; // ignora self-loops

        int du = (int)boost::degree(u, G);
        int dv = (int)boost::degree(v, G);
        if (use_excess) {
            du = std::max(0, du - 1);
            dv = std::max(0, dv - 1);
        }

        S1 += (double)du * (double)dv;
        S2 += 0.5 * ((double)du + (double)dv);
        S3 += 0.5 * ((double)du * du + (double)dv * dv);
        ++M_use;
    }

    if (M_use == 0) { // só tinha self-loop
        out.R = 0.0;
        out.error = 0.0;
        return out;
    }

    const double M = (double)M_use;
    const double T1 = S1 / M, T2 = S2 / M, T3 = S3 / M;
    const double denom = T3 - T2 * T2;

    if (std::abs(denom) < 1e-12) {
        out.R = 0.0;        // todos graus iguais no conjunto usado
        out.error = 0.0;
        return out;
    }

    const double r_full = (T1 - T2 * T2) / denom;
    out.R = r_full;

    // ======================= DAGJK por blocos ==========================
    // bucket de cada vértice
    std::vector<int> bucket_of_vertex(Nz, 0);
    for (auto v : make_iterator_range(boost::vertices(G))) {
        std::size_t id = (std::size_t)vidx[v];
        bucket_of_vertex[id] = (int)(mix_hash(id) % (std::size_t)B);
    }

    std::vector<double> r_blocks; r_blocks.reserve(B);
    double r_bar = 0.0;
    int B_eff = 0;

    // Para cada bloco b: remove todas as arestas incidentes a vértices do bloco
    for (int b = 0; b < B; ++b) {
        double s1 = 0.0, s2 = 0.0, s3 = 0.0;
        std::size_t m_sub = 0;

        for (auto e : make_iterator_range(boost::edges(G))) {
            auto u = boost::source(e, G);
            auto v = boost::target(e, G);
            if (u == v) continue; // ignora self-loop

            std::size_t iu = (std::size_t)vidx[u];
            std::size_t iv = (std::size_t)vidx[v];
            if (bucket_of_vertex[iu] == b || bucket_of_vertex[iv] == b) {
                continue; // remove aresta que toca o bloco b
            }

            // graus devem ser reavaliados NO SUBGRAFO restante.
            // Para manter custo linear, aproximamos usando os graus do grafo original
            // menos as incidências ao bloco b. Para exatidão (ainda O(M)), contamos de novo:
            // 1) Primeiro passei direto aos somatórios usando contagem local:

            // Truque: computar "on the fly" os graus efetivos no subgrafo é caro sem cache.
            // Portanto, fazemos DUAS passadas: (1) contar graus no subgrafo, (2) somatórios.
        }

        // ---------- Passo 1: contar graus no subgrafo ----------
        // (recontagem linear O(M)): deg_tmp[id] = grau do vértice id sem arestas do bloco b
        std::vector<int> deg_tmp(Nz, 0);
        for (auto e : make_iterator_range(boost::edges(G))) {
            auto u = boost::source(e, G);
            auto v = boost::target(e, G);
            if (u == v) continue;

            std::size_t iu = (std::size_t)vidx[u];
            std::size_t iv = (std::size_t)vidx[v];
            if (bucket_of_vertex[iu] == b || bucket_of_vertex[iv] == b) continue;

            deg_tmp[iu] += 1;
            deg_tmp[iv] += 1;
        }

        // ---------- Passo 2: somatórios T1,T2,T3 no subgrafo ----------
        for (auto e : make_iterator_range(boost::edges(G))) {
            auto u = boost::source(e, G);
            auto v = boost::target(e, G);
            if (u == v) continue;

            std::size_t iu = (std::size_t)vidx[u];
            std::size_t iv = (std::size_t)vidx[v];
            if (bucket_of_vertex[iu] == b || bucket_of_vertex[iv] == b) continue;

            int du = deg_tmp[iu];
            int dv = deg_tmp[iv];
            if (use_excess) {
                du = std::max(0, du - 1);
                dv = std::max(0, dv - 1);
            }

            s1 += (double)du * (double)dv;
            s2 += 0.5 * ((double)du + (double)dv);
            s3 += 0.5 * ((double)du * du + (double)dv * dv);
            ++m_sub;
        }

        if (m_sub == 0) continue;

        const double m  = (double)m_sub;
        const double t1 = s1 / m, t2 = s2 / m, t3 = s3 / m;
        const double den = t3 - t2 * t2;
        if (std::abs(den) < 1e-12) continue;

        const double r_b = (t1 - t2 * t2) / den;
        r_blocks.push_back(r_b);
        r_bar += r_b;
        ++B_eff;
    }

    if (B_eff >= 2) {
        r_bar /= (double)B_eff;
        double var = 0.0;
        for (double rb : r_blocks) {
            const double d = rb - r_bar;
            var += d * d;
        }
        // DAGJK: (B_eff - 1)/B_eff * sum(...)
        var *= (double)(B_eff - 1) / (double)B_eff;
        out.error = std::sqrt(std::max(0.0, var));
    } else {
        out.error = 0.0; // não deu para estimar com blocos válidos
    }

    return out;
}

// ---- helpers (iguais/compatíveis com os seus) ----
static std::vector<double> compute_ranks_average_ties(const std::vector<double>& vals) {
    int n = (int)vals.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return vals[a] < vals[b];
    });
    std::vector<double> ranks(n);
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && vals[idx[j]] == vals[idx[i]]) ++j;
        double rank_sum = 0.0;
        for (int t = i; t < j; ++t) rank_sum += (t + 1); // ranks 1..n
        double avg_rank = rank_sum / double(j - i);
        for (int t = i; t < j; ++t) ranks[idx[t]] = avg_rank;
        i = j;
    }
    return ranks;
}

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
        sx  += dx * dx;
        sy  += dy * dy;
    }
    double denom = std::sqrt(sx * sy);
    if (denom <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return num / denom;
}

// Spearman a partir de listas de graus nas extremidades (um par por aresta)
static double spearman_from_degree_pairs(const std::vector<double>& deg_u_list,
                                         const std::vector<double>& deg_v_list)
{
    if (deg_u_list.empty() || deg_u_list.size() != deg_v_list.size()) return 0.0;
    auto ranks_u = compute_ranks_average_ties(deg_u_list);
    auto ranks_v = compute_ranks_average_ties(deg_v_list);
    double rho = pearson_correlation(ranks_u, ranks_v);
    if (std::isnan(rho)) rho = 0.0;
    if (rho > 1.0 && rho < 1.0 + 1e-12) rho = 1.0;
    if (rho < -1.0 && rho > -1.0 - 1e-12) rho = -1.0;
    return rho;
}


/**
 * Spearman por ranks com erro via DAGJK (delete-a-group jackknife) por blocos de vértices.
 * @param B           número de blocos (ex.: 100). Use B ∈ [50, 200] para bons resultados.
 * @param use_excess  se true, usa excess degree (k-1), com piso 0.
 */
R_ass_Spearman SamuraI::computeRankAssortativitySpearmanDAGJK(int B, bool use_excess /*=false*/) {
    using boost::make_iterator_range;

    R_ass_Spearman out;

    const std::size_t Mz = boost::num_edges(G);
    const std::size_t Nz = boost::num_vertices(G);
    if (Mz == 0 || Nz == 0) return out;

    if (B < 2) B = 2; // precisa de pelo menos 2 blocos para variância

    // property map para obter índice [0..N-1] de cada vértice (requer vecS ou vertex_index instalado)
    auto vidx = get(boost::vertex_index, G);

    // ---------- rho no grafo completo ----------
    std::vector<double> deg_u_full; deg_u_full.reserve(Mz);
    std::vector<double> deg_v_full; deg_v_full.reserve(Mz);

    for (auto e : make_iterator_range(boost::edges(G))) {
        auto u = boost::source(e, G);
        auto v = boost::target(e, G);
        if (u == v) continue; // ignora self-loops
        int ku = static_cast<int>(boost::degree(u, G));
        int kv = static_cast<int>(boost::degree(v, G));
        if (use_excess) {
            ku = std::max(0, ku - 1);
            kv = std::max(0, kv - 1);
        }
        deg_u_full.push_back((double)ku);
        deg_v_full.push_back((double)kv);
    }

    double rho_full = spearman_from_degree_pairs(deg_u_full, deg_v_full);
    out.R = rho_full;

    // ---------- DAGJK por blocos de vértices ----------
    // Pré-computa o "bucket" de cada vértice
    std::vector<int> bucket_of_vertex(Nz, 0);
    for (auto v : make_iterator_range(boost::vertices(G))) {
        std::size_t id = (std::size_t)vidx[v];
        bucket_of_vertex[id] = (int)(mix_hash(id) % (std::size_t)B);
    }

    std::vector<double> rho_block; rho_block.reserve(B);
    double rho_bar = 0.0;
    int B_eff = 0;

    // Vetor de graus temporários (recontados no subgrafo restante)
    std::vector<int> deg_tmp(Nz, 0);

    for (int b = 0; b < B; ++b) {
        // Zera graus temporários
        std::fill(deg_tmp.begin(), deg_tmp.end(), 0);

        // 1) Conta graus no subgrafo removendo todas as arestas que tocam vértices do bloco b
        std::size_t M_sub = 0;
        for (auto e : make_iterator_range(boost::edges(G))) {
            auto u = boost::source(e, G);
            auto v = boost::target(e, G);
            std::size_t iu = (std::size_t)vidx[u];
            std::size_t iv = (std::size_t)vidx[v];

            if (bucket_of_vertex[iu] == b || bucket_of_vertex[iv] == b) {
                continue; // descarta aresta que toca bloco b
            }
            if (u == v) continue; // ignora self-loop

            // conta grau no subgrafo
            deg_tmp[iu] += 1;
            deg_tmp[iv] += 1;
            ++M_sub;
        }

        if (M_sub == 0) {
            // bloco muito grande; não há arestas restantes — ignora bloco
            continue;
        }

        // 2) Monta listas de graus nos extremos das arestas restantes e calcula Spearman
        std::vector<double> deg_u_list; deg_u_list.reserve(M_sub);
        std::vector<double> deg_v_list; deg_v_list.reserve(M_sub);

        for (auto e : make_iterator_range(boost::edges(G))) {
            auto u = boost::source(e, G);
            auto v = boost::target(e, G);
            std::size_t iu = (std::size_t)vidx[u];
            std::size_t iv = (std::size_t)vidx[v];

            if (bucket_of_vertex[iu] == b || bucket_of_vertex[iv] == b) {
                continue;
            }
            if (u == v) continue;

            int ku = deg_tmp[iu];
            int kv = deg_tmp[iv];
            if (use_excess) {
                ku = std::max(0, ku - 1);
                kv = std::max(0, kv - 1);
            }
            deg_u_list.push_back((double)ku);
            deg_v_list.push_back((double)kv);
        }

        if (deg_u_list.empty()) continue;

        double rho_b = spearman_from_degree_pairs(deg_u_list, deg_v_list);
        rho_block.push_back(rho_b);
        rho_bar += rho_b;
        ++B_eff;
    }

    if (B_eff >= 2) {
        rho_bar /= (double)B_eff;

        double var = 0.0;
        for (double rb : rho_block) {
            double d = rb - rho_bar;
            var += d * d;
        }
        // DAGJK: (B_eff - 1) / B_eff * sum(...)
        var *= (double)(B_eff - 1) / (double)B_eff;
        out.error = std::sqrt(std::max(0.0, var));
    } else {
        // não deu para estimar variância com blocos válidos suficientes
        out.error = 0.0;
    }

    return out;
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

