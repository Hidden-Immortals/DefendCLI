import os
import re
import json
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import detector
import igraph as ig
import leidenalg
import logging
from multiprocessing import Pool
import traceback

# Use a non-interactive backend to generate images without a display environment
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def read_focus_data(directory):
    """
    Reads data files from the specified directory, yielding JSON objects.
    It filters files to only process those with specific naming patterns and handles errors.
    """
    for filename in os.listdir(directory):
        # Filter to read specific files
        if not filename.endswith('.gz') and 'ta1-cadets-e3-official' in filename:
            filepath = os.path.join(directory, filename)
            logging.info(f"Reading file: {filepath}")
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    for each in file:
                        yield json.loads(each)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {filename}: {e}")
            except Exception as e:
                logging.error(f"An error occurred while processing {filename}: {e}")

def CADETS_provenance_graph(data_iterator, batch_size=1000):
    """
    Constructs a provenance graph from provided data. Processes data in batches for efficiency.
    """
    G = nx.DiGraph()
    seen_cmdLines = set()
    data_batch = []

    def process_batch(batch, G, seen_cmdLines):
        """
        Processes a batch of data to construct and update the graph.
        """
        subjects = {}
        # First pass: process all Subject data
        for each in batch:
            try:
                first_key, first_value = next(iter(each['datum'].items()))
                if first_key == 'com.bbn.tc.schema.avro.cdm18.Subject':
                    subject = each['datum'][first_key]
                    startTimestampNanos = subject.get('startTimestampNanos')
                    cid = subject.get('cid')
                    if startTimestampNanos is not None:
                        subjects[startTimestampNanos] = cid
            except (KeyError, TypeError, StopIteration) as e:
                logging.error(f"Error processing subject data: {e}")

        # Second pass: process all Event data
        for each in batch:
            try:
                first_key, first_value = next(iter(each['datum'].items()))
                if first_key == 'com.bbn.tc.schema.avro.cdm18.Event':
                    event = each['datum'][first_key]
                    ppid = event['properties']['map'].get('ppid')
                    cmdLine = event['properties']['map'].get('cmdLine')
                    event_timestampNanos = event.get('timestampNanos')

                    # Check if event's timestamp is in subjects
                    if event_timestampNanos in subjects:
                        node_id = subjects[event_timestampNanos]
                        if cmdLine:
                            cmdLine_str = str(cmdLine)
                            if cmdLine_str not in seen_cmdLines:
                                seen_cmdLines.add(cmdLine_str)
                                G.add_node(node_id, cmdLine=cmdLine_str)
                            if ppid is not None:
                                if not G.has_node(ppid):
                                    G.add_node(ppid)
                                if not G.has_edge(ppid, node_id):
                                    G.add_edge(ppid, node_id)
            except (KeyError, TypeError, StopIteration) as e:
                logging.error(f"Error processing event data: {e}")

        # Third pass: add combined properties
        for each in batch:
            try:
                first_key, first_value = next(iter(each['datum'].items()))
                if first_key == 'com.bbn.tc.schema.avro.cdm18.Event':
                    event = each['datum'][first_key]
                    event_timestampNanos = event.get('timestampNanos')
                    event_type_nework = ['EVENT_RECVFROM','EVENT_ACCEPT','EVENT_CONNECT','EVENT_SENDTO']
                    if event_timestampNanos in subjects:
                        node_id = subjects[event_timestampNanos]
                        event_type = event.get('type')
                        if event_type in event_type_nework:
                            IP = event['properties']['map'].get('address')
                            port = event['properties']['map'].get('port')
                            if IP and port:
                                if node_id in G.nodes:
                                    G.nodes[node_id]['network_info'] = {'type': event_type, 'IP': IP, 'port': port}
            except (KeyError, TypeError, StopIteration) as e:
                logging.error(f"Error adding network properties: {e}")
                logging.error(f"Event data: {event}")
                traceback.print_exc()

    for each in data_iterator:
        data_batch.append(each)
        if len(data_batch) == batch_size:
            process_batch(data_batch, G, seen_cmdLines)
            data_batch.clear()

    # Process remaining data
    if data_batch:
        process_batch(data_batch, G, seen_cmdLines)

    return G

def provenance_graph(data):
    """
    Constructs a provenance graph from data. Each node represents a command line invocation.
    """
    G = nx.DiGraph()
    seen_cmdLines = set()
    for each in data:
        first_key, first_value = next(iter(each['datum'].items()))
        if 'ppid' in str(each['datum'][first_key]):
            if first_key == 'com.bbn.tc.schema.avro.cdm18.Subject':
                ppid = each['datum'][first_key]['properties']['map']['ppid']
                cid = each['datum'][first_key]['cid']
                cmdLine = each['datum'][first_key]['cmdLine']
                cmdLine_str = str(cmdLine)
                if cmdLine and cmdLine_str not in seen_cmdLines:
                    seen_cmdLines.add(cmdLine_str)
                    G.add_node(cid, cmdLine=cmdLine_str)
                    if not G.has_node(ppid):
                        G.add_node(ppid)
                if not G.has_edge(ppid, cid):
                    G.add_edge(ppid, cid)
    return G

def directed_acyclic_graph(G):
    """
    Converts the given graph to a directed acyclic graph by removing cycles.
    """
    if nx.is_directed_acyclic_graph(G):
        return G
    while not nx.is_directed_acyclic_graph(G):
        edge_list = list(nx.find_cycle(G, orientation='original'))
        G.remove_edges_from(edge_list)
    return G

def remove_empty_cmd_subgraphs(G):
    """
    Removes subgraphs from the graph that do not contain any nodes with command line data.
    """
    subgraphs = list(nx.weakly_connected_components(G))
    for subgraph in subgraphs:
        all_empty = True
        for node in subgraph:
            if 'cmdLine' in G.nodes[node] and G.nodes[node]['cmdLine']:
                all_empty = False
                break
        if all_empty:
            G.remove_nodes_from(subgraph)
    return G

def pagerank_worker(subgraph_edges, subgraph_nodes, max_iter):
    """
    Worker function for computing PageRank in parallel.
    """
    logging.info(f"Starting pagerank worker for subgraph with {len(subgraph_edges)} edges")
    subgraph = nx.DiGraph()
    subgraph.add_edges_from(subgraph_edges)
    nx.set_node_attributes(subgraph, subgraph_nodes)
    result = nx.pagerank(subgraph, max_iter=max_iter)
    logging.info(f"Finished pagerank worker for subgraph with {len(subgraph_edges)} edges")
    return result

def betweenness_worker(G, nodes):
    """
    Worker function for computing betweenness centrality in parallel.
    """
    logging.info(f"Starting betweenness worker for {len(nodes)} nodes")
    result = nx.betweenness_centrality_subset(G, sources=nodes, targets=nodes, normalized=True)
    logging.info(f"Finished betweenness worker for {len(nodes)} nodes")
    return result

def calculate_pagerank(G, max_iter=100, num_cores=16):
    """
    Calculates the PageRank of nodes in the graph using parallel processing.
    """
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

def calculate_edge_weight_wrapper(args):
    """
    Wrapper function for calculating edge weight to be used in parallel processing.
    """
    edge, page_rank, betweenness_centrality = args
    return calculate_edge_weight(edge, page_rank, betweenness_centrality)

def calculate_edge_weight(edge, page_rank, betweenness_centrality):
    """
    Calculates the weight of an edge based on PageRank and betweenness centrality.
    """
    u, v = edge
    PR_sum = page_rank[u] + page_rank[v]
    CB_sum = betweenness_centrality[u] + betweenness_centrality[v]
    return PR_sum + CB_sum

def parallel_betweenness_centrality(G, num_cores=16):
    """
    Calculates betweenness centrality of nodes in the graph using parallel processing.
    """
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

def weight_assignment(G, max_iter=100, num_cores=16):
    """
    Assigns weights to the edges of the graph based on PageRank and betweenness centrality.
    """
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

def refine_weight_linux(G):
    """
    Refines edge weights in the graph based on Linux command line data.
    """
    with open('cmd_linux.json', 'r') as f:
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

def find_sources(G):
    """
    Identifies source nodes in the graph (nodes with no incoming edges).
    """
    logging.info("Finding sources...")
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    logging.info(f"Found {len(sources)} sources.")
    return sources

def find_sinks(G):
    """
    Identifies sink nodes in the graph (nodes with no outgoing edges).
    """
    logging.info("Finding sinks...")
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    logging.info(f"Found {len(sinks)} sinks.")
    return sinks

def find_sources_and_sinks(G, num_cores=16):
    """
    Identifies source and sink nodes in the graph using parallel processing.
    """
    with Pool(num_cores) as pool:
        sources_result = pool.apply_async(find_sources, (G,))
        sinks_result = pool.apply_async(find_sinks, (G,))
        sources = sources_result.get()
        sinks = sinks_result.get()
    logging.info(f"Found {len(sources)} sources and {len(sinks)} sinks.")
    return sources, sinks

def convert_to_igraph(G):
    """
    Converts a NetworkX graph to an iGraph graph for community detection.
    """
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G_igraph = ig.Graph(directed=True)
    G_igraph.add_vertices(len(G.nodes()))
    G_igraph.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])
    return G_igraph, mapping

def leiden_community_detection(G):
    """
    Performs community detection on the graph using the Leiden algorithm.
    """
    G_igraph, mapping = convert_to_igraph(G)
    partitions = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
    community_map = {list(mapping.keys())[list(mapping.values()).index(v)]: p for v, p in
                     enumerate(partitions.membership)}
    return community_map

def calculate_community_importance(G, community_map):
    """
    Calculates the importance of each community based on edge weights.
    """
    community_importance = {}
    for node, community in community_map.items():
        if community not in community_importance:
            community_importance[community] = 0
        community_importance[community] += sum(G[node][neighbor].get('weight', 1) for neighbor in G.neighbors(node))
    return community_importance

def compute_shortest_path_serial_2(G, sources, sinks, community_map, community_importance):
    """
    Computes shortest paths in the graph considering community importance, in serial.
    """
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

    if shortest_paths:
        max_community_score = max(score for _, (_, _, score) in shortest_paths.items())
    else:
        max_community_score = 1

    for key in shortest_paths:
        length, path, community_score = shortest_paths[key]
        normalized_community_score = community_score / max_community_score
        adjusted_length = length / (1 + normalized_community_score)
        shortest_paths[key] = (adjusted_length, path, normalized_community_score)

    sorted_paths = sorted(shortest_paths.items(), key=lambda x: (-x[1][2], -len(x[1][1]), x[1][0]))
    logging.info(f"Finished optimized shortest path computation in serial. Found {len(sorted_paths)} paths")
    return sorted_paths

def compute_shortest_path_serial(G, sources, sinks, community_map, community_importance):
    """
    Computes shortest paths in the graph considering community importance, in serial using Dijkstra's algorithm.
    """
    logging.info(f"Starting Dijkstra all pairs with community detection in serial")
    shortest_paths = {}
    for source in sources:
        for sink in sinks:
            try:
                length, path = nx.single_source_dijkstra(G, source, target=sink)
                if all(G.nodes[node].get('cmdLine', '') == '' for node in path):
                    continue
                path_communities = [community_map[node] for node in path if node in community_map]
                if path_communities:
                    community_score = sum(community_importance[comm] for comm in path_communities)
                    shortest_paths[(source, sink)] = (length, path, community_score)
            except nx.NetworkXNoPath:
                continue

    if shortest_paths:
        max_community_score = max(score for _, (_, _, score) in shortest_paths.items())
    else:
        max_community_score = 1

    for key in shortest_paths:
        length, path, community_score = shortest_paths[key]
        normalized_community_score = community_score / max_community_score
        adjusted_length = length / (1 + normalized_community_score)
        shortest_paths[key] = (adjusted_length, path, normalized_community_score)

    sorted_paths = sorted(shortest_paths.items(), key=lambda x: (-x[1][2], -len(x[1][1]), x[1][0]))
    logging.info(f"Finished Dijkstra all pairs with community detection in serial. Found {len(sorted_paths)} paths")
    return sorted_paths

def graph_visualization(G):
    """
    Visualizes the graph and saves the output as an image.
    """
    pos = nx.spring_layout(G)
    degrees = dict(G.degree())
    node_size = [v * 5 for v in degrees.values()]
    font_size = {node: min((degree + 1) * 2, 0.1) for node, degree in degrees.items()}
    node_labels = {node: f"{node}\nET: {G.nodes[node].get('cmdLine', 'N/A')}" for node in G.nodes()}
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=node_size, node_color='skyblue', edge_color='gray', width=0.2)
    for node, (x, y) in pos.items():
        plt.text(x, y, node_labels[node], fontsize=font_size[node], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.savefig('networkx_graph.png', dpi=720)

def write_results_to_file(results, filename='results.json'):
    """
    Writes the results to a JSON file.
    """
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    logging.info('Read data...')
    directory_path = '/root/Engagement-3/data/cadets' # Change File Path here
    data = read_focus_data(directory_path)  # Read data
    logging.info('Graph build...')
    G = CADETS_provenance_graph(data, batch_size=10000)  # Build provenance graph
    logging.info('Graph compression...')
    DAG = directed_acyclic_graph(G)  # Convert to directed acyclic graph
    DAG = remove_empty_cmd_subgraphs(DAG)  # Remove empty command line subgraphs
    logging.info('Weight assignment...')
    G_weighted = weight_assignment(DAG, max_iter=100, num_cores=16)  # Use 16-core CPU
    logging.info('Refine weight...')
    re_G_weighted = refine_weight_linux(G_weighted)  # Refine weights
    logging.info('Leiden community detection...')
    community_map = leiden_community_detection(re_G_weighted)  # Community detection
    community_importance = calculate_community_importance(re_G_weighted,
                                                          community_map)  # Calculate community importance
    logging.info('InfoPath Search...')
    sources, sinks = find_sources_and_sinks(re_G_weighted, num_cores=16)  # Find sources and sinks
    sorted_shortest_paths = compute_shortest_path_serial_2(re_G_weighted, sources, sinks, community_map,
                                                         community_importance)
    logging.info('Detection...')
    result = detector.run(re_G_weighted, sorted_shortest_paths)  # Run detector
    write_results_to_file(result)  # Write results to file
    logging.info('Results written to results.json')
