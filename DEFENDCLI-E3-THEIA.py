import os
import re
import json
import math
import networkx as nx
import matplotlib
import DEFENDCLI_DETECTOR as detector
import igraph as ig
import leidenalg
import logging
from multiprocessing import Pool
import traceback
import ujson
import mmap
from concurrent.futures import ThreadPoolExecutor

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# --- Function Definitions ---

def read_focus_data(directory, num_workers=8):
    """
    Read and parse JSON files in parallel using multithreading.
    - directory: path to directory containing trace JSON files.
    - num_workers: number of threads to use for parallel file processing.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if
             not f.endswith('.gz') and 'ta1-theia-e3-official' in f]

    def process_file(filepath):
        try:
            logging.info(f"Reading file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    for line in mmapped_file.read().decode("utf-8").splitlines():
                        try:
                            yield ujson.loads(line)
                        except ujson.JSONDecodeError as e:
                            logging.error(f"JSON decode error in {filepath}: {e}")
                        except Exception as e:
                            logging.error(f"Unexpected error in {filepath}: {e}")
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_data in executor.map(process_file, files):
            yield from file_data

def provenance_graph(data):
    """
    Build a provenance graph from parsed data using cmdLine as node identifiers.
    Retain only nodes with the highest process ID per unique command line.
    """
    G = nx.DiGraph()
    cmdline_map = {}

    for each in data:
        first_key, first_value = next(iter(each['datum'].items()))
        if 'ppid' in str(each['datum'][first_key]):
            if first_key == 'com.bbn.tc.schema.avro.cdm18.Subject':
                ppid = int(each['datum'][first_key]['properties']['map']['ppid'])
                cid = int(each['datum'][first_key]['cid'])
                cmdLine = each['datum'][first_key]['cmdLine']
                cmdLine_str = str(cmdLine) if cmdLine else None

                if cmdLine_str:
                    if cmdLine_str in cmdline_map:
                        if cid > cmdline_map[cmdLine_str][0]:
                            cmdline_map[cmdLine_str] = (cid, ppid)
                    else:
                        cmdline_map[cmdLine_str] = (cid, ppid)

    for cmdLine_str, (cid, ppid) in cmdline_map.items():
        G.add_node(cid, cmdLine=cmdLine_str)
        if not G.has_node(ppid):
            G.add_node(ppid)
        G.add_edge(ppid, cid)

    nodes_to_remove = [node for node in list(G.nodes) if G.nodes[node].get('cmdLine') is None and G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)

    return G

def directed_acyclic_graph(G):
    """
    Ensure the graph is a Directed Acyclic Graph (DAG) by removing cycles.
    """
    if nx.is_directed_acyclic_graph(G):
        return G
    while not nx.is_directed_acyclic_graph(G):
        edge_list = list(nx.find_cycle(G, orientation='original'))
        G.remove_edges_from(edge_list)
    return G

def pagerank_worker(subgraph_edges, subgraph_nodes, max_iter):
    """
    Compute PageRank for a subgraph.
    """
    subgraph = nx.DiGraph()
    subgraph.add_edges_from(subgraph_edges)
    nx.set_node_attributes(subgraph, subgraph_nodes)
    return nx.pagerank(subgraph, max_iter=max_iter)

def betweenness_worker(G, nodes):
    """
    Compute betweenness centrality for a subset of nodes.
    """
    return nx.betweenness_centrality_subset(G, sources=nodes, targets=nodes, normalized=True)

def calculate_pagerank(G, max_iter=100, num_cores=16):
    """
    Compute PageRank in parallel for each weakly connected component.
    """
    subgraphs = [(list(G.subgraph(c).edges()), dict(G.subgraph(c).nodes(data=True))) for c in
                 nx.connected_components(G.to_undirected())]
    pagerank_results = {}
    with Pool(num_cores) as pool:
        results = pool.starmap(pagerank_worker, [(e, n, max_iter) for e, n in subgraphs])
    for result in results:
        pagerank_results.update(result)
    return pagerank_results

def calculate_edge_weight(edge, page_rank, betweenness):
    """
    Calculate weight for an edge based on PageRank and betweenness.
    """
    u, v = edge
    return page_rank[u] + page_rank[v] + betweenness[u] + betweenness[v]

def calculate_edge_weight_wrapper(args):
    edge, page_rank, betweenness = args
    return calculate_edge_weight(edge, page_rank, betweenness)

def parallel_betweenness_centrality(G, num_cores=16):
    """
    Compute betweenness centrality in parallel for all nodes.
    """
    nodes = list(G.nodes())
    chunk_size = max(1, len(nodes) // num_cores)
    chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    betweenness = dict.fromkeys(G, 0.0)
    with Pool(num_cores) as pool:
        results = pool.starmap(betweenness_worker, [(G, chunk) for chunk in chunks])
    for result in results:
        for node, value in result.items():
            betweenness[node] += value
    return betweenness

def weight_assignment(G, max_iter=100, num_cores=16):
    """
    Assign edge weights using PageRank and betweenness centrality.
    Normalize and invert weights for shortest path use.
    """
    for (u, v) in G.edges():
        G[u][v]['weight'] = 1

    page_rank = calculate_pagerank(G, max_iter, num_cores)
    betweenness = parallel_betweenness_centrality(G, num_cores)
    args = [(e, page_rank, betweenness) for e in G.edges()]
    with Pool(num_cores) as pool:
        weights = pool.map(calculate_edge_weight_wrapper, args)

    min_weight = min(weights)
    max_weight = 9999999999
    max_diff = max(weights) - min_weight
    decimal_places = 0 if max_diff == 0 else -int(math.floor(math.log10(max_diff))) + 1

    for (u, v), w in zip(G.edges(), weights):
        w = round(w, decimal_places)
        G[u][v]['weight'] = max_weight if w == min_weight else 1 / max(w, 1e-10)

    return G

def refine_weight_linux(G):
    """
    Adjust weights based on Linux-specific suspicious command patterns.
    """
    with open('cmd_linux.json', 'r') as f:
        cmd_linux_data = json.load(f)

    level_weights = {"High": 2.5, "Medium": 2, "Low": 1.5}

    for node in G.nodes(data=True):
        cmd = node[1].get('cmdLine', '')
        for act in cmd_linux_data['activities']:
            regex = act['attack_signature']['regex']
            level = act['Level']
            if re.search(regex, str(cmd)):
                for nbr in G.neighbors(node[0]):
                    G[node[0]][nbr]['weight'] /= level_weights[level]
                for pred in G.predecessors(node[0]):
                    G[pred][node[0]]['weight'] /= level_weights[level]
    return G

def find_sources(G):
    return [n for n in G.nodes if G.in_degree(n) == 0]

def find_sinks(G):
    return [n for n in G.nodes if G.out_degree(n) == 0]

def find_sources_and_sinks(G, num_cores=16):
    with Pool(num_cores) as pool:
        s = pool.apply_async(find_sources, (G,))
        t = pool.apply_async(find_sinks, (G,))
        return s.get(), t.get()

def convert_to_igraph(G):
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(len(G.nodes()))
    G_ig.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])
    return G_ig, mapping

def leiden_community_detection(G):
    G_ig, mapping = convert_to_igraph(G)
    parts = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    return {list(mapping.keys())[list(mapping.values()).index(i)]: c for i, c in enumerate(parts.membership)}

def calculate_community_importance(G, community_map):
    importance = {}
    for node, com in community_map.items():
        importance.setdefault(com, 0)
        importance[com] += sum(G[node][nbr].get('weight', 1) for nbr in G.neighbors(node))
    return importance

def compute_shortest_path(G, sources, sinks, community_map, community_importance):
    """
    Compute shortest paths from sources to sinks with community importance as a factor.
    """
    paths = {}
    for s in sources:
        spaths = nx.shortest_path(G, source=s, weight='weight')
        lengths = nx.shortest_path_length(G, source=s, weight='weight')
        for t in sinks:
            if t not in spaths: continue
            path = spaths[t]
            length = lengths[t]
            if all(G.nodes[n].get('cmdLine', '') == '' for n in path): continue
            comm_score = sum(community_importance.get(community_map.get(n, -1), 0) for n in path)
            paths[(s, t)] = (length, path, comm_score)

    max_score = max((s for _, (_, _, s) in paths.items()), default=1)
    for k in paths:
        l, p, cs = paths[k]
        norm_cs = cs / max_score
        paths[k] = (l / (1 + norm_cs), p, norm_cs)

    return sorted(paths.items(), key=lambda x: (-x[1][2], -len(x[1][1]), x[1][0]))

def compute_all_paths(G, community_map, community_importance):
    paths = {}
    for s in G.nodes():
        spaths = nx.shortest_path(G, source=s, weight='weight')
        lengths = nx.shortest_path_length(G, source=s, weight='weight')
        for t, path in spaths.items():
            if s == t: continue
            length = lengths[t]
            if all(G.nodes[n].get('cmdLine', '') == '' for n in path): continue
            comm_score = sum(community_importance.get(community_map.get(n, -1), 0) for n in path)
            paths[(s, t)] = (length, path, comm_score)

    max_score = max((s for _, (_, _, s) in paths.items()), default=1)
    for k in paths:
        l, p, cs = paths[k]
        norm_cs = cs / max_score
        paths[k] = (l / (1 + norm_cs), p, norm_cs)

    return sorted(paths.items(), key=lambda x: (-x[1][2], -len(x[1][1]), x[1][0]))

def write_results_to_file(results, filename='results.json'):
    """
    Save final detection results to a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

# --- Main Script ---

if __name__ == '__main__':
    try:
        logging.info('Reading data...')
        directory_path = '/root/Engagement-3/data/theia'
        data = read_focus_data(directory_path, num_workers=8)

        logging.info('Building graph...')
        G = provenance_graph(data)
        print(f"Graph G has {len(G.nodes())} nodes and {len(G.edges())} edges.")

        logging.info('Compressing graph...')
        DAG = directed_acyclic_graph(G)
        print(f"DAG has {len(DAG.nodes())} nodes and {len(DAG.edges())} edges.")

        logging.info('Assigning weights...')
        weighted_G = weight_assignment(DAG, max_iter=100, num_cores=16)

        logging.info('Refining weights...')
        refined_weighted_G = refine_weight_linux(weighted_G)

        logging.info('Detecting communities...')
        community_map = leiden_community_detection(refined_weighted_G)
        community_importance = calculate_community_importance(refined_weighted_G, community_map)

        logging.info('Searching InfoPaths...')
        sources, sinks = find_sources_and_sinks(refined_weighted_G, num_cores=16)

        try:
            sorted_paths = compute_shortest_path(refined_weighted_G, sources, sinks, community_map, community_importance)
        except Exception as e:
            logging.error(f"Shortest path calculation failed: {e}")
            logging.info("Falling back to compute all paths...")
            sorted_paths = compute_all_paths(refined_weighted_G, community_map, community_importance)

        logging.info('Running detection...')
        result = detector.run(refined_weighted_G, sorted_paths)
        write_results_to_file(result)
        logging.info('Results saved to results.json')

        total_edges = len(refined_weighted_G.edges())
        nonzero_weight_edges = [(u, v, d['weight']) for u, v, d in refined_weighted_G.edges(data=True) if d.get('weight', 0) != 0]
        nonzero_weight_count = len(nonzero_weight_edges)
        sorted_edges = sorted(nonzero_weight_edges, key=lambda x: x[2])

        print(f"Total edges: {total_edges}")
        print(f"Edges with non-zero weight: {nonzero_weight_count}")
        print("Top 10 edges by weight:")
        for edge in sorted_edges[:10]:
            print(edge)

    except Exception as main_exception:
        logging.error(f"Fatal error: {main_exception}")
        logging.error(traceback.format_exc())
