# -*- coding: utf-8 -*-
import re
import json
import math
import networkx as nx
import DEFENDCLI_DETECTOR as detector
import igraph as ig
import leidenalg
import logging
from multiprocessing import Pool

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Read and parse JSON-formatted log data from a file
def read_focus_data(path):
    logging.info(f"Reading file: {path}")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while processing {path}: {e}")
        return None

# Parse log message into a dictionary
def parse_message(message):
    lines = message.split('\r\n')
    parsed_dict = {}
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
        else:
            key = line.rstrip(':')
            value = None
        if key:
            parsed_dict[key] = value
    return parsed_dict

# Construct a provenance graph from the parsed data
def provenance_graph(data):
    G = nx.DiGraph()
    seen_command_lines = set()

    for log in data:
        process_id = log.get('ProcessId', '')
        command_line = log.get('CommandLine', '')
        parent_process_id = log.get('ParentProcessId', '')
        parent_command_line = log.get('ParentCommandLine', '')

        if process_id and parent_process_id:
            if not G.has_node(process_id):
                G.add_node(process_id, cmdLine=set(), network_info=[])
            if command_line and command_line not in seen_command_lines:
                G.nodes[process_id]['cmdLine'].add(command_line)
                seen_command_lines.add(command_line)
            if not G.has_node(parent_process_id):
                G.add_node(parent_process_id, cmdLine=set(), network_info=[])
            if parent_command_line and parent_command_line not in seen_command_lines:
                G.nodes[parent_process_id]['cmdLine'].add(parent_command_line)
                seen_command_lines.add(parent_command_line)
            if not G.has_edge(parent_process_id, process_id):
                G.add_edge(parent_process_id, process_id)

        if 'SourceIp' in log and 'DestinationIp' in log and G.has_node(process_id):
            network_info = {
                'source_ip': log.get('SourceIp', ''),
                'source_port': log.get('SourcePort', ''),
                'destination_ip': log.get('DestinationIp', ''),
                'destination_port': log.get('DestinationPort', '')
            }
            G.nodes[process_id]['network_info'].append(network_info)

        if 'QueryName' in log and G.has_node(process_id):
            network_info = {'query_name': log.get('QueryName', '')}
            G.nodes[process_id]['network_info'].append(network_info)

    return G

# Ensure the graph is a directed acyclic graph (DAG) by removing cycles
def directed_acyclic_graph(G):
    if nx.is_directed_acyclic_graph(G):
        return G
    while not nx.is_directed_acyclic_graph(G):
        edge_list = list(nx.find_cycle(G, orientation='original'))
        G.remove_edges_from(edge_list)
    return G

# Print each connected component of the graph
def print_connected_components(G):
    connected_components = list(nx.connected_components(G.to_undirected()))
    for i, component in enumerate(connected_components, 1):
        print(f"Connected component {i}: {component}")

# PageRank worker for multiprocessing
def pagerank_worker(subgraph_edges, subgraph_nodes, max_iter):
    logging.info(f"Starting pagerank worker for subgraph with {len(subgraph_edges)} edges")
    subgraph = nx.DiGraph()
    subgraph.add_edges_from(subgraph_edges)
    nx.set_node_attributes(subgraph, subgraph_nodes)
    result = nx.pagerank(subgraph, max_iter=max_iter)
    logging.info(f"Finished pagerank worker for subgraph with {len(subgraph_edges)} edges")
    return result

# Betweenness centrality worker for multiprocessing
def betweenness_worker(G, nodes):
    logging.info(f"Starting betweenness worker for {len(nodes)} nodes")
    result = nx.betweenness_centrality_subset(G, sources=nodes, targets=nodes, normalized=True)
    logging.info(f"Finished betweenness worker for {len(nodes)} nodes")
    return result

# Parallel PageRank calculation
def calculate_pagerank(G, max_iter=100, num_cores=16):
    subgraphs = [(list(G.subgraph(c).edges()), dict(G.subgraph(c).nodes(data=True))) for c in
                 nx.connected_components(G.to_undirected())]
    pagerank_results = {}
    with Pool(num_cores) as pool:
        results = pool.starmap(pagerank_worker,
                               [(subgraph_edges, subgraph_nodes, max_iter) for subgraph_edges, subgraph_nodes in
                                subgraphs])
    for result in results:
        pagerank_results.update(result)
    return pagerank_results

# Wrapper for edge weight calculation with arguments
def calculate_edge_weight_wrapper(args):
    edge, page_rank, betweenness_centrality = args
    return calculate_edge_weight(edge, page_rank, betweenness_centrality)

# Compute edge weight from PageRank and betweenness centrality
def calculate_edge_weight(edge, page_rank, betweenness_centrality):
    u, v = edge
    PR_sum = page_rank[u] + page_rank[v]
    CB_sum = betweenness_centrality[u] + betweenness_centrality[v]
    return PR_sum + CB_sum

# Parallel betweenness centrality computation
def parallel_betweenness_centrality(G, num_cores=16):
    nodes = list(G.nodes())
    chunk_size = len(nodes) // num_cores
    chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    betweenness = dict.fromkeys(G, 0.0)
    with Pool(num_cores) as pool:
        results = pool.starmap(betweenness_worker, [(G, chunk) for chunk in chunks])
    for result in results:
        for node, value in result.items():
            betweenness[node] += value
    return betweenness

# Assign weights to graph edges using PageRank and betweenness centrality
def weight_assignment(G, max_iter=100, num_cores=16):
    for (u, v) in G.edges():
        G[u][v]['weight'] = 1
    page_rank = calculate_pagerank(G, max_iter=max_iter, num_cores=num_cores)
    betweenness_centrality = parallel_betweenness_centrality(G, num_cores=num_cores)
    args = [(edge, page_rank, betweenness_centrality) for edge in G.edges()]
    with Pool(num_cores) as pool:
        weights = pool.map(calculate_edge_weight_wrapper, args)
    min_weight = min(weights)
    max_weight = 9999999999
    max_diff = max(weights) - min_weight
    best_decimal_places = 0 if max_diff == 0 else -int(math.floor(math.log10(max_diff))) + 1
    for (u, v), weight in zip(G.edges(), weights):
        rounded_weight = round(weight, best_decimal_places)
        if rounded_weight == 0:
            rounded_weight = 1e-10
        G[u][v]['weight'] = max_weight if rounded_weight == min_weight else 1 / rounded_weight
    return G

# Adjust edge weights based on command-line patterns for Windows
def refine_weight_windows(G):
    with open('cmd_windows.json', 'r') as f:
        cmd_linux_data = json.load(f)
    level_weights = {"High": 2.5, "Medium": 2, "Low": 1.5}
    for node in G.nodes(data=True):
        cmd_line = node[1].get('cmdLine', '')
        for activity in cmd_linux_data['activities']:
            regex = activity['attack_signature']['regex']
            level = activity['Level']
            if re.search(regex, str(cmd_line)):
                edges_to_adjust = False
                for neighbor in G.neighbors(node[0]):
                    if G.has_edge(node[0], neighbor):
                        edges_to_adjust = True
                        current_weight = G[node[0]][neighbor].get('weight', 1)
                        new_weight = current_weight / level_weights[level]
                        G[node[0]][neighbor]['weight'] = new_weight
                for predecessor in G.predecessors(node[0]):
                    if G.has_edge(predecessor, node[0]):
                        edges_to_adjust = True
                        current_weight = G[predecessor][node[0]].get('weight', 1)
                        new_weight = current_weight / level_weights[level]
                        G[predecessor][node[0]]['weight'] = new_weight
                if edges_to_adjust:
                    logging.info('Regex: %s', regex)
                    logging.info('Find CMD: %s', cmd_line)
                    logging.info('*********')
    return G

# Identify source nodes (zero in-degree)
def find_sources(G):
    logging.info("Finding sources...")
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    logging.info(f"Found {len(sources)} sources.")
    return sources

# Identify sink nodes (zero out-degree)
def find_sinks(G):
    logging.info("Finding sinks...")
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    logging.info(f"Found {len(sinks)} sinks.")
    return sinks

# Find both sources and sinks in parallel
def find_sources_and_sinks(G, num_cores=16):
    with Pool(num_cores) as pool:
        sources_result = pool.apply_async(find_sources, (G,))
        sinks_result = pool.apply_async(find_sinks, (G,))
        sources = sources_result.get()
        sinks = sinks_result.get()
    logging.info(f"Found {len(sources)} sources and {len(sinks)} sinks.")
    return sources, sinks

# Convert NetworkX graph to iGraph format for Leiden algorithm
def convert_to_igraph(G):
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G_igraph = ig.Graph(directed=True)
    G_igraph.add_vertices(len(G.nodes()))
    G_igraph.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])
    return G_igraph, mapping

# Perform Leiden community detection on the graph
def leiden_community_detection(G):
    G_igraph, mapping = convert_to_igraph(G)
    partitions = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
    community_map = {list(mapping.keys())[list(mapping.values()).index(v)]: p for v, p in enumerate(partitions.membership)}
    return community_map

# Calculate community importance by summing weights of outgoing edges in the community
def calculate_community_importance(G, community_map):
    community_importance = {}
    for node, community in community_map.items():
        if community not in community_importance:
            community_importance[community] = 0
        community_importance[community] += sum(G[node][neighbor].get('weight', 1) for neighbor in G.neighbors(node))
    return community_importance

# Compute shortest paths from sources to sinks and score them
def compute_shortest_path(G, sources, sinks, community_map, community_importance):
    logging.info(f"Starting optimized shortest path computation in serial")
    shortest_paths = {}
    for source in sources:
        paths = nx.shortest_path(G, source=source, weight='weight')
        lengths = nx.shortest_path_length(G, source=source, weight='weight')
        for sink in sinks:
            if sink not in paths:
                continue
            path = paths[sink]
            length = lengths[sink]
            if all(G.nodes[node].get('cmdLine', '') == '' for node in path):
                continue
            path_communities = [community_map.get(node, -1) for node in path]
            community_score = sum(community_importance.get(comm, 0) for comm in path_communities if comm != -1)
            shortest_paths[(source, sink)] = (length, path, community_score)

    max_community_score = max((score for _, (_, _, score) in shortest_paths.items()), default=1)

    for key in shortest_paths:
        length, path, community_score = shortest_paths[key]
        normalized_community_score = community_score / max_community_score
        adjusted_length = length / (1 + normalized_community_score)
        shortest_paths[key] = (adjusted_length, path, normalized_community_score)

    sorted_paths = sorted(shortest_paths.items(), key=lambda x: (-x[1][2], -len(x[1][1]), x[1][0]))
    logging.info(f"Finished optimized shortest path computation in serial. Found {len(sorted_paths)} paths")
    return sorted_paths

# Compute all possible paths between nodes with community-based scoring
def compute_all_paths(G, community_map, community_importance):
    logging.info(f"Starting computation of all paths")
    all_paths = {}
    for source in G.nodes():
        paths = nx.shortest_path(G, source=source, weight='weight')
        lengths = nx.shortest_path_length(G, source=source, weight='weight')
        for target, path in paths.items():
            if source == target:
                continue
            length = lengths[target]
            if all(G.nodes[node].get('cmdLine', '') == '' for node in path):
                continue
            path_communities = [community_map.get(node, -1) for node in path]
            community_score = sum(community_importance.get(comm, 0) for comm in path_communities if comm != -1)
            all_paths[(source, target)] = (length, path, community_score)

    max_community_score = max((score for _, (_, _, score) in all_paths.items()), default=1)

    for key in all_paths:
        length, path, community_score = all_paths[key]
        normalized_community_score = community_score / max_community_score
        adjusted_length = length / (1 + normalized_community_score)
        all_paths[key] = (adjusted_length, path, normalized_community_score)

    sorted_paths = sorted(all_paths.items(), key=lambda x: (-x[1][2], -len(x[1][1]), x[1][0]))
    logging.info(f"Finished computation of all paths. Found {len(sorted_paths)} paths")
    return sorted_paths

# Write final results to a JSON file
def write_results_to_file(results, filename='results.json'):
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

# Main execution logic
if __name__ == '__main__':
    logging.info('Read data...')
    path = '/root/attack_scenario_3.json'
    data = read_focus_data(path)
    message = [parse_message(each['Message']) for each in data]

    logging.info('Load data...')
    G = provenance_graph(message)

    logging.info('Graph compression...')
    DAG = directed_acyclic_graph(G)

    logging.info('Weight assignment...')
    G_weighted = weight_assignment(DAG, max_iter=100, num_cores=16)

    logging.info('Refine weight...')
    re_G_weighted = refine_weight_windows(G_weighted)

    logging.info('Leiden community detection...')
    community_map = leiden_community_detection(re_G_weighted)
    community_importance = calculate_community_importance(re_G_weighted, community_map)

    logging.info('InfoPath Search...')
    sources, sinks = find_sources_and_sinks(re_G_weighted, num_cores=16)
    try:
        sorted_paths = compute_shortest_path(re_G_weighted, sources, sinks, community_map, community_importance)
    except Exception as e:
        logging.error(f"Shortest path calculation failed: {e}")
        logging.info("Falling back to compute all paths...")
        sorted_paths = compute_all_paths(re_G_weighted, community_map, community_importance)

    logging.info('Detection...')
    result = detector.run(re_G_weighted, sorted_paths)
    write_results_to_file(result)
    logging.info('Results written to results.json')

    total_edges = len(re_G_weighted.edges())
    nonzero_weight_edges = [(u, v, d['weight']) for u, v, d in re_G_weighted.edges(data=True) if d.get('weight', 0) != 0]
    nonzero_weight_count = len(nonzero_weight_edges)
    sorted_edges_by_weight = sorted(nonzero_weight_edges, key=lambda x: x[2])

    print(f"Total edges: {total_edges}")
    print(f"Edges with non-zero weights: {nonzero_weight_count}")
    print("Top 10 edges by weight:")
    for edge in sorted_edges_by_weight[:10]:
        print(edge)