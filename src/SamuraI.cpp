#include "SamuraI.hpp"
#include "eigen_vec.hpp"
#include "randUtils.hpp"

#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/graph/breadth_first_search.hpp> //BFS
#include <boost/graph/visitors.hpp>
#include <boost/array.hpp>
#include<zlib.h>
#include <limits>

inline bool cmp(vertex_prop_double p1, vertex_prop_double p2) { return p1.second < p2.second; }

void SamuraI::reportProgress(std::size_t completed, std::size_t total) const {
    if (!progress_callback || total == 0) return;
    const std::size_t interval = std::max<std::size_t>(1, total / 10);
    if (completed == total || completed % interval == 0)
        progress_callback(completed, total);
}

void SamuraI::CreateNetwork(){
    hrand rnd(m_seed);
    const int m0 = m_m0;
    if (m0 < 1) throw std::runtime_error("m_m0 deve ser >= 1");
    if (m_num_vertices < m0 + 1)
        throw std::runtime_error("Para grau minimo m0, exija m_num_vertices >= m0 + 1");

    clear();
    center_of_mass.setZero();
    sum_positions.setZero();
    pos.reserve(static_cast<std::size_t>(m_num_vertices));
    // Only selected vertices change degree during the m0 draws, and they
    // are excluded thereafter. All remaining attachment weights stay fixed.
    std::vector<double> weights(static_cast<std::size_t>(m_num_vertices));
    std::vector<unsigned char> chosen(static_cast<std::size_t>(m_num_vertices), 0);
    const double exponent = -0.5 * m_alpha_a;

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
            add_weighted_edge(u, v);
        }
    }
    reportProgress(m_init, m_num_vertices);

    // 2) Crescimento
    for (int i = m_init; i < m_num_vertices; ++i){
        boost::add_vertex(G);
        add_pos(i);
        vertex_t New = i;

        std::fill(chosen.begin(), chosen.begin() + i, 0);
        double initial_total = 0.0;
        for (int u = 0; u < i; ++u) {
            const double degree = static_cast<double>(boost::degree(u, G));
            if (m_alpha_a == 0.0) {
                // pow(r_squared, 0) is exactly 1: geometry has no effect.
                weights[u] = degree;
            } else {
                const Vector4d Ruv = pos[u] - pos[New];
                double Ruv_SQ = Ruv.transpose() * Ruv;
                if (Ruv_SQ <= 0.0) Ruv_SQ = std::numeric_limits<double>::min();
                weights[u] = degree * std::pow(Ruv_SQ, exponent);
            }
            initial_total += weights[u];
        }

        for (int ell = 0; ell < m0; ++ell) {
            // Sum in the original vertex order: subtracting selected weights
            // from a running total would change floating-point rounding.
            double p_total = initial_total;
            if (ell != 0) {
                p_total = 0.0;
                for (int u = 0; u < i; ++u) {
                    if (!chosen[u]) p_total += weights[u];
                }
            }

            if (p_total <= 0.0) {
                int trials = 0;
                while (trials++ < 1000) {
                    const int v = rnd.uniform_int(0, i - 1);
                    if (!chosen[v]) {
                        add_weighted_edge(v, New);
                        chosen[v] = 1;
                        break;
                    }
                }
                continue;
            }

            const double r = rnd.uniform_real(0.0, 1.0);
            double cumsum = 0.0;
            for (int u = 0; u < i; ++u) {
                if (chosen[u]) continue;
                cumsum += weights[u] / p_total;
                if (cumsum > r) {
                    add_weighted_edge(u, New);
                    chosen[u] = 1;
                    break;
                }
            }
        }
        reportProgress(i + 1, m_num_vertices);
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


bool SamuraI::computeTreeNavigation(Navigation_BFS& bfs, Navigation_COST& cost) const {
    const std::size_t n = boost::num_vertices(G);
    if (n == 0 || boost::num_edges(G) != n - 1) return false;

    const auto weights = get(boost::edge_weight, G);
    std::vector<vertex_t> parent(n, n), order;
    std::vector<double> parent_weight(n, 0.0);
    order.reserve(n);
    order.push_back(0);
    parent[0] = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        const vertex_t u = order[i];
        for (auto e : boost::make_iterator_range(boost::out_edges(u, G))) {
            const vertex_t v = boost::target(e, G);
            if (!std::isfinite(weights[e])) return false;
            if (parent[v] != n) continue;
            parent[v] = u;
            parent_weight[v] = weights[e];
            order.push_back(v);
        }
    }
    if (order.size() != n) return false;

    std::vector<std::size_t> subtree(n, 1);
    std::vector<int> height(n, 0);
    std::vector<double> weighted_height(n, 0.0);
    long double hop_sum = 0.0L, weight_sum = 0.0L;
    for (std::size_t i = n; i-- > 1;) {
        const vertex_t v = order[i], u = parent[v];
        // Removing this edge partitions the tree into s and n-s vertices.
        // Exactly s*(n-s) unordered paths cross the edge.
        const long double pairs = static_cast<long double>(subtree[v]) * (n - subtree[v]);
        hop_sum += pairs;
        weight_sum += pairs * parent_weight[v];
        subtree[u] += subtree[v];
        const int branch = height[v] + 1;
        bfs.diamater = std::max(bfs.diamater, height[u] + branch);
        height[u] = std::max(height[u], branch);
        const double weighted_branch = weighted_height[v] + parent_weight[v];
        cost.diamater = std::max(cost.diamater, weighted_height[u] + weighted_branch);
        weighted_height[u] = std::max(weighted_height[u], weighted_branch);
    }
    const long double pairs = static_cast<long double>(n) * (n - 1) / 2;
    bfs.shortestpath = pairs ? static_cast<double>(hop_sum / pairs) : 0.0;
    cost.shortestpath = pairs ? static_cast<double>(weight_sum / pairs) : 0.0;
    return true;
}

Navigation_BFS SamuraI::computeGlobalNavigation_BFS() {
    Navigation_BFS result;
    Navigation_COST cost;
    if (computeTreeNavigation(result, cost)) return result;
    const std::size_t n = boost::num_vertices(G);
    if (n <= 1) return result;

    // Flatten adjacency once for all sources. Reuse the queue and distance
    // buffers instead of allocating visitor/color/queue storage for every BFS.
    std::vector<std::size_t> offsets(n + 1, 0);
    std::vector<vertex_t> neighbors;
    neighbors.reserve(2 * boost::num_edges(G));
    for (vertex_t u = 0; u < n; ++u) {
        for (auto v : boost::make_iterator_range(boost::adjacent_vertices(u, G)))
            neighbors.push_back(v);
        offsets[u + 1] = neighbors.size();
    }
    std::vector<int> distance(n);
    std::vector<vertex_t> queue(n);
    double sum = 0.0;
    for (vertex_t source = 0; source < n; ++source) {
        std::fill(distance.begin(), distance.end(), -1);
        distance[source] = 0;
        queue[0] = source;
        std::size_t tail = 1;
        for (std::size_t head = 0; head < tail; ++head) {
            const vertex_t u = queue[head];
            for (std::size_t j = offsets[u]; j < offsets[u + 1]; ++j) {
                const vertex_t v = neighbors[j];
                if (distance[v] != -1) continue;
                distance[v] = distance[u] + 1;
                queue[tail++] = v;
            }
        }
        for (int d : distance) {
            // Match the original zero contribution from unreachable vertices.
            if (d < 0) continue;
            sum += d;
            result.diamater = std::max(result.diamater, d);
        }
        reportProgress(source + 1, n);
    }
    result.shortestpath = sum / (static_cast<double>(n) * (n - 1));
    return result;
}

Navigation_BFS SamuraI::computeGlobalNavigation_BFSSampled(std::size_t sample_size) {
    Navigation_BFS result;
    const std::size_t n = boost::num_vertices(G);
    if (n <= 1 || sample_size == 0) return result;

    // Sampling sources uniformly without replacement makes the average of
    // their distances to all other vertices an unbiased estimator of the
    // all-pairs mean shortest-path length.
    sample_size = std::min(sample_size, n);
    std::vector<vertex_t> sources(n);
    std::iota(sources.begin(), sources.end(), vertex_t{0});
    std::mt19937 sampler(static_cast<std::mt19937::result_type>(m_seed));
    std::shuffle(sources.begin(), sources.end(), sampler);
    sources.resize(sample_size);

    // As in the complete BFS calculation, build the adjacency storage once
    // and reuse queue and distance buffers for every sampled source.
    std::vector<std::size_t> offsets(n + 1, 0);
    std::vector<vertex_t> neighbors;
    neighbors.reserve(2 * boost::num_edges(G));
    for (vertex_t u = 0; u < n; ++u) {
        for (auto v : boost::make_iterator_range(boost::adjacent_vertices(u, G)))
            neighbors.push_back(v);
        offsets[u + 1] = neighbors.size();
    }

    std::vector<int> distance(n);
    std::vector<vertex_t> queue(n);
    long double sum = 0.0L;
    for (std::size_t source_index = 0; source_index < sample_size; ++source_index) {
        const vertex_t source = sources[source_index];
        std::fill(distance.begin(), distance.end(), -1);
        distance[source] = 0;
        queue[0] = source;
        std::size_t tail = 1;
        for (std::size_t head = 0; head < tail; ++head) {
            const vertex_t u = queue[head];
            for (std::size_t j = offsets[u]; j < offsets[u + 1]; ++j) {
                const vertex_t v = neighbors[j];
                if (distance[v] != -1) continue;
                distance[v] = distance[u] + 1;
                queue[tail++] = v;
            }
        }
        for (const int d : distance) {
            // Preserve the full-BFS convention: unreachable vertices add zero
            // while remaining in the estimator's N - 1 denominator.
            if (d < 0) continue;
            sum += d;
            result.diamater = std::max(result.diamater, d);
        }
        reportProgress(source_index + 1, sample_size);
    }
    result.shortestpath = static_cast<double>(sum /
        (static_cast<long double>(sample_size) * static_cast<long double>(n - 1)));
    return result;
}


void SamuraI::add_weighted_edge(int u, int v) {
    auto [e, ok] = boost::add_edge(u, v, G);
    if (!ok) return;
    const Vector4d d  = pos[u] - pos[v];
    const double w2   = (d.transpose() * d);
    const double w    = std::sqrt(std::max(0.0, w2)); // distância física
    put(boost::edge_weight, G, e, w);
}


static inline int clamp_int(int x, int lo, int hi){ return std::max(lo, std::min(hi, x)); }

#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <limits>
#include <cmath>

Navigation_COST SamuraI::computeGlobalNavigationDijkstraExact(void) {
    using boost::make_iterator_range;

    Navigation_COST out;

    const std::size_t N = boost::num_vertices(G);
    if (N <= 1) {
        out.shortestpath     = 0.0;
        out.diamater         = 0.0;
        out.shortestpath_se  = 0.0;
        out.coverage         = 1.0;
        return out;
    }

    Navigation_BFS bfs;
    if (computeTreeNavigation(bfs, out)) return out;

    // A contiguous, immutable adjacency layout avoids following allocations
    // in the growing graph for every relaxation of every source. Represent
    // each undirected edge by two arcs, preserving the outgoing edge order.
    using CompactGraph = boost::compressed_sparse_row_graph<
        boost::directedS, boost::no_property,
        boost::property<boost::edge_weight_t, double>>;
    CompactGraph compact(G, get(boost::vertex_index, G), N, 2 * boost::num_edges(G));
    auto wmap = get(boost::edge_weight, compact);
    const auto original_weights = get(boost::edge_weight, G);
    for (vertex_t u = 0; u < N; ++u) {
        auto compact_edge = boost::out_edges(u, compact).first;
        for (auto e : make_iterator_range(boost::out_edges(u, G))) {
            wmap[*compact_edge++] = original_weights[e];
        }
    }
    auto vindex = get(boost::vertex_index, compact);
    auto colors = boost::make_two_bit_color_map(N, vindex);

    // Buffer de distância
    std::vector<double> dist(N, std::numeric_limits<double>::infinity());
    auto dist_map = boost::make_iterator_property_map(dist.begin(), vindex);

    // Acumuladores globais
    long double   sum_all = 0.0L;     // soma das distâncias mínimas (usar long double p/ estabilidade)
    std::uint64_t cnt_all = 0;        // nº de pares alcançáveis (ordenados u->v, u!=v)
    double        diam    = 0.0;      // diâmetro (máx. das distâncias mínimas)

    // Loop por TODAS as fontes (exato)
    for (vertex_t s : make_iterator_range(boost::vertices(G))) {
        boost::dijkstra_shortest_paths(
            compact, s,
            boost::weight_map(wmap)
                 .distance_map(dist_map)
                 .color_map(colors)
        );

        // Acumula resultados desta fonte
        for (vertex_t v : make_iterator_range(boost::vertices(G))) {
            if (v == s) continue;
            const double dv = dist[v];
            if (std::isfinite(dv)) {
                sum_all += dv;
                ++cnt_all;
                if (dv > diam) diam = dv;
            }
        }
        reportProgress(s + 1, N);
    }

    // Média exata sobre pares alcançáveis
    out.shortestpath    = (cnt_all ? static_cast<double>(sum_all / (long double)cnt_all) : 0.0);
    out.diamater        = diam;
    out.shortestpath_se = 0.0; // exato, sem erro amostral

    // Cobertura = fração de pares alcançáveis entre N*(N-1)
    const long double total_pairs = static_cast<long double>(N) * static_cast<long double>(N - 1);
    out.coverage = (total_pairs > 0.0L ? static_cast<double>((long double)cnt_all / total_pairs) : 1.0);

    return out;
}



double SamuraI::computeClusterCoefficient() {
    const std::size_t n = boost::num_vertices(G);
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();

    // Orient each edge by (degree, vertex id). Every triangle has exactly one
    // forward wedge, avoiding pairwise searches in high-degree neighborhoods.
    std::vector<std::vector<vertex_t>> forward(n);
    std::vector<std::size_t> degree(n), triangles(n, 0);
    for (vertex_t v = 0; v < n; ++v) degree[v] = boost::degree(v, G);
    for (auto e : boost::make_iterator_range(boost::edges(G))) {
        vertex_t u = boost::source(e, G), v = boost::target(e, G);
        if (std::make_pair(degree[v], v) < std::make_pair(degree[u], u)) std::swap(u, v);
        forward[u].push_back(v);
    }
    std::vector<vertex_t> marked(n, n);
    for (vertex_t u = 0; u < n; ++u) {
        for (vertex_t v : forward[u]) marked[v] = u;
        for (vertex_t v : forward[u]) {
            for (vertex_t w : forward[v]) {
                if (marked[w] != u) continue;
                ++triangles[u];
                ++triangles[v];
                ++triangles[w];
            }
        }
        reportProgress(u + 1, n);
    }

    // Boost's original ClusteringContainer used float, including accumulation.
    // Keep that precision and vertex order so output rounding is unchanged.
    float sum = 0.0f;
    for (vertex_t v = 0; v < n; ++v) {
        const float routes = static_cast<float>(degree[v] * (degree[v] - 1) / 2);
        const float cc = routes > 0.0f ? static_cast<float>(triangles[v]) / routes : 0.0f;
        sum += cc;
    }
    return sum / static_cast<float>(n);
}

static std::vector<std::pair<vertex_t, vertex_t>> edge_endpoints(const graph_t& graph) {
    std::vector<std::pair<vertex_t, vertex_t>> endpoints;
    endpoints.reserve(boost::num_edges(graph));
    for (auto e : boost::make_iterator_range(boost::edges(graph))) {
        const vertex_t u = boost::source(e, graph), v = boost::target(e, graph);
        if (u != v) endpoints.emplace_back(u, v);
    }
    return endpoints;
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

    R_ass_Newman out{};

    const std::size_t Mz = boost::num_edges(G);
    const std::size_t Nz = boost::num_vertices(G);
    if (Mz == 0 || Nz == 0) return out;
    if (B < 2) B = 2;

    auto vidx = get(boost::vertex_index, G);
    const auto endpoints = edge_endpoints(G);

    // ======================= r no grafo completo =======================
    double S1 = 0.0, S2 = 0.0, S3 = 0.0; // somatórios (não normalizados)
    std::size_t M_use = 0;

    for (const auto& [u, v] : endpoints) {
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

    std::vector<int> deg_tmp(Nz, 0);

    // Para cada bloco b: remove todas as arestas incidentes a vértices do bloco
    for (int b = 0; b < B; ++b) {
        double s1 = 0.0, s2 = 0.0, s3 = 0.0;
        std::size_t m_sub = 0;

        // ---------- Passo 1: contar graus no subgrafo ----------
        // (recontagem linear O(M)): deg_tmp[id] = grau do vértice id sem arestas do bloco b
        std::fill(deg_tmp.begin(), deg_tmp.end(), 0);
        for (const auto& [u, v] : endpoints) {
            if (u == v) continue;

            std::size_t iu = (std::size_t)vidx[u];
            std::size_t iv = (std::size_t)vidx[v];
            if (bucket_of_vertex[iu] == b || bucket_of_vertex[iv] == b) continue;

            deg_tmp[iu] += 1;
            deg_tmp[iv] += 1;
        }

        // ---------- Passo 2: somatórios T1,T2,T3 no subgrafo ----------
        for (const auto& [u, v] : endpoints) {
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

        if (m_sub == 0) {
            reportProgress(b + 1, B);
            continue;
        }

        const double m  = (double)m_sub;
        const double t1 = s1 / m, t2 = s2 / m, t3 = s3 / m;
        const double den = t3 - t2 * t2;
        if (std::abs(den) < 1e-12) {
            reportProgress(b + 1, B);
            continue;
        }

        const double r_b = (t1 - t2 * t2) / den;
        r_blocks.push_back(r_b);
        r_bar += r_b;
        ++B_eff;
        reportProgress(b + 1, B);
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
// Inputs are nonnegative integer degrees, so a histogram gives the same
// average ranks for ties without sorting O(M log M) values per jackknife block.
static std::vector<double> compute_ranks_average_ties(const std::vector<double>& vals) {
    if (vals.empty()) return {};
    const auto max_degree = static_cast<std::size_t>(*std::max_element(vals.begin(), vals.end()));
    std::vector<std::size_t> counts(max_degree + 1, 0);
    for (double value : vals) ++counts[static_cast<std::size_t>(value)];
    std::vector<double> rank_by_degree(counts.size(), 0.0);
    std::size_t offset = 0;
    for (std::size_t k = 0; k < counts.size(); ++k) {
        if (counts[k] == 0) continue;
        double rank_sum = 0.0;
        for (std::size_t j = 0; j < counts[k]; ++j) rank_sum += offset + j + 1;
        rank_by_degree[k] = rank_sum / static_cast<double>(counts[k]);
        offset += counts[k];
    }
    std::vector<double> ranks;
    ranks.reserve(vals.size());
    for (double value : vals) ranks.push_back(rank_by_degree[static_cast<std::size_t>(value)]);
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

    R_ass_Spearman out{};

    const std::size_t Mz = boost::num_edges(G);
    const std::size_t Nz = boost::num_vertices(G);
    if (Mz == 0 || Nz == 0) return out;

    if (B < 2) B = 2; // precisa de pelo menos 2 blocos para variância

    // property map para obter índice [0..N-1] de cada vértice (requer vecS ou vertex_index instalado)
    auto vidx = get(boost::vertex_index, G);
    const auto endpoints = edge_endpoints(G);

    // ---------- rho no grafo completo ----------
    std::vector<double> deg_u_full; deg_u_full.reserve(Mz);
    std::vector<double> deg_v_full; deg_v_full.reserve(Mz);

    for (const auto& [u, v] : endpoints) {
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
    auto& deg_u_list = deg_u_full;
    auto& deg_v_list = deg_v_full;

    for (int b = 0; b < B; ++b) {
        // Zera graus temporários
        std::fill(deg_tmp.begin(), deg_tmp.end(), 0);

        // 1) Conta graus no subgrafo removendo todas as arestas que tocam vértices do bloco b
        std::size_t M_sub = 0;
        for (const auto& [u, v] : endpoints) {
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
            reportProgress(b + 1, B);
            continue;
        }

        // 2) Monta listas de graus nos extremos das arestas restantes e calcula Spearman
        deg_u_list.clear();
        deg_v_list.clear();

        for (const auto& [u, v] : endpoints) {
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

        if (deg_u_list.empty()) {
            reportProgress(b + 1, B);
            continue;
        }

        double rho_b = spearman_from_degree_pairs(deg_u_list, deg_v_list);
        rho_block.push_back(rho_b);
        rho_bar += rho_b;
        ++B_eff;
        reportProgress(b + 1, B);
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
