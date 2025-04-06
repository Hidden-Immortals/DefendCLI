import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
import json
import re
from joblib import Parallel, delayed
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Preprocess text by lowercasing and removing non-word characters
def preprocess_text(text):
    '''
    Preprocess the input text by converting to lowercase and removing non-word characters.
    Returns a list of tokens.
    '''
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.split()

# Compute SimHash from a vector
def simhash(vec, bits=64):
    '''
    Compute a SimHash fingerprint for a given vector.
    The hash is reduced to a specified number of bits (default: 64).
    '''
    if len(vec) == 0:
        return '0' * bits
    v = np.zeros(bits)
    for component in vec:
        hash_val = int(hash(component)) & ((1 << bits) - 1)
        binary_str = bin(hash_val)[2:].zfill(bits)
        for i in range(bits):
            if binary_str[i] == '1':
                v[i] += 1
            else:
                v[i] -= 1
    return ''.join(['1' if x > 0 else '0' for x in v])

# Compute Hamming distance between two hashes
def hamming_distance(hash1, hash2):
    '''
    Compute the Hamming distance between two binary hash strings.
    '''
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# Train a Word2Vec model on given texts
def train_word2vec(texts, vector_size=60, window=5, min_count=1, workers=4):
    '''
    Train a Word2Vec model using preprocessed texts.
    '''
    processed_texts = [preprocess_text(text) for text in texts]
    model = Word2Vec(sentences=processed_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# Train a FastText model on given texts
def train_fasttext(texts, vector_size=60, window=5, min_count=1, workers=4):
    '''
    Train a FastText model using preprocessed texts.
    '''
    processed_texts = [preprocess_text(text) for text in texts]
    model = FastText(sentences=processed_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# Train a Doc2Vec model on given texts
def train_doc2vec(texts, vector_size=60, epochs=20, workers=4):
    '''
    Train a Doc2Vec model using preprocessed texts.
    '''
    tagged_data = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(texts)]
    model = Doc2Vec(vector_size=vector_size, epochs=epochs, workers=workers)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Convert a text to vector using a word vector model
def text_to_vector(text, model):
    '''
    Convert a single text into a vector by averaging its word embeddings from a trained model.
    '''
    words = preprocess_text(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Convert a text to vector using a Doc2Vec model
def text_to_vector_doc2vec(text, model):
    '''
    Infer a vector representation for a single document using a trained Doc2Vec model.
    '''
    return model.infer_vector(preprocess_text(text))

# Compute SimHash for a list of texts using a given model
def calculate_simhash_for_texts(texts, model, doc2vec_model=False):
    '''
    Compute SimHash values and vector representations for each text using a specified model.
    '''
    if doc2vec_model:
        vectors = [text_to_vector_doc2vec(text, model) for text in texts]
    else:
        vectors = [text_to_vector(text, model) for text in texts]
    simhashes = [simhash(vector) for vector in vectors]
    return simhashes, vectors

# Evaluate how many SimHash pairs are similar below a Hamming distance threshold
def evaluate_simhashes(simhashes, threshold=10):
    '''
    Evaluate the ratio of SimHash pairs whose Hamming distance is below the given threshold.
    '''
    if len(simhashes) == 0:
        return 0
    simhash_array = np.array([[int(bit) for bit in hash_str] for hash_str in simhashes], dtype=np.uint8)
    dist_matrix = squareform(pdist(simhash_array, metric='hamming')) * simhash_array.shape[1]
    distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    similar_count = np.sum(distances <= threshold)
    return similar_count / len(distances) if distances.size > 0 else 0

# Perform hard-voting anomaly detection using multiple models
def hard_voting_anomaly_detection(vectors, contamination=0.5):
    '''
    Use an ensemble of anomaly detection models to perform hard-voting based outlier detection.
    Returns the indices of data points considered anomalous by a majority of models.
    '''
    vectors = np.array(vectors)
    if np.isnan(vectors).any():
        print("\u26a0\ufe0f Warning: Found NaN values in vectors! Replacing with mean values.")
        col_means = np.nanmean(vectors, axis=0)
        inds = np.where(np.isnan(vectors))
        vectors[inds] = np.take(col_means, inds[1])

    models = {
        'IsolationForest': IsolationForest(n_estimators=16, max_samples=0.5, contamination=contamination, random_state=42),
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=16, contamination=contamination, n_jobs=-1),
        'OneClassSVM': OneClassSVM(nu=contamination, kernel='rbf', gamma='scale'),
        'EllipticEnvelope': EllipticEnvelope(contamination=contamination, support_fraction=1, random_state=42),
        'KMeans': KMeans(n_clusters=16, random_state=42, n_init=20, max_iter=300),
        'GaussianMixture': GaussianMixture(n_components=16, covariance_type='full', max_iter=200, random_state=42)
    }

    def fit_predict_model(name, model, vectors):
        '''
        Helper function to fit a model and return its anomaly predictions.
        '''
        try:
            if name == 'LocalOutlierFactor':
                predictions = model.fit_predict(vectors)
            else:
                model.fit(vectors)
                predictions = model.predict(vectors)
            return name, predictions
        except ValueError as e:
            print(f"\u274c Error with model {name}: {e}")
            return name, np.ones(len(vectors))

    results = Parallel(n_jobs=-1)(delayed(fit_predict_model)(name, model, vectors) for name, model in models.items())

    predictions = {}
    for name, pred in results:
        if name in ['KMeans', 'GaussianMixture']:
            pred = np.where(pred == 0, 1, -1)
        if name == 'LocalOutlierFactor':
            pred = np.where(pred == 1, 1, -1)
        predictions[name] = pred

    combined_anomalies = sum((pred == -1).astype(int) for pred in predictions.values())
    final_anomalies = np.where(combined_anomalies >= 4)[0]

    return final_anomalies

# Convert cmdLine field to string if it's a set
def convert_cmdLine_to_str(cmdLine):
    '''
    Convert a set of command-line tokens to a single space-separated string.
    '''
    if isinstance(cmdLine, set):
        return ' '.join(cmdLine)
    return cmdLine

# SimHash-based recommender for vector generation
def SimHash_Recommender(G, sorted_shortest_paths):
    '''
    Generate text representations of paths in the graph, train multiple models,
    compute SimHash vectors, and return the most consistent set of vectors.
    '''
    texts = [
        '\n'.join([convert_cmdLine_to_str(G.nodes[node].get('cmdLine', '') or 'N/A') for node in path])
        for _, (_, path, _) in sorted_shortest_paths
    ]

    print("Training Word2Vec model...")
    word2vec_model = train_word2vec(texts)

    print("Training FastText model...")
    fasttext_model = train_fasttext(texts)

    print("Training Doc2Vec model...")
    doc2vec_model = train_doc2vec(texts)

    print("Calculating SimHash for Word2Vec...")
    word2vec_simhashes, word2vec_vectors = calculate_simhash_for_texts(texts, word2vec_model)

    print("Calculating SimHash for Doc2Vec...")
    doc2vec_simhashes, doc2vec_vectors = calculate_simhash_for_texts(texts, doc2vec_model, doc2vec_model=True)

    print("Calculating SimHash for FastText...")
    fasttext_simhashes, fasttext_vectors = calculate_simhash_for_texts(texts, fasttext_model)

    print("Evaluating SimHash distances...")
    word2vec_score = evaluate_simhashes(word2vec_simhashes)
    doc2vec_score = evaluate_simhashes(doc2vec_simhashes)
    fasttext_score = evaluate_simhashes(fasttext_simhashes)

    print(f"Word2Vec Score: {word2vec_score}")
    print(f"Doc2Vec Score: {doc2vec_score}")
    print(f"FastText Score: {fasttext_score}")

    max_score = max(word2vec_score, doc2vec_score, fasttext_score)
    if max_score == word2vec_score:
        print("Word2Vec has the maximum score.")
        best_vectors = word2vec_vectors
    elif max_score == doc2vec_score:
        print("Doc2Vec has the maximum score.")
        best_vectors = doc2vec_vectors
    else:
        print("FastText has the maximum score.")
        best_vectors = fasttext_vectors

    return best_vectors, texts

# Print and save anomalous paths as a graph to JSON
def print_anomalous_graph(anomalous_texts, output_file="infopath.json"):
    '''
    Build a subgraph from anomalous texts and paths, annotate with metadata, and save as JSON.
    '''
    G = nx.DiGraph()
    subgraph_data = []

    for item in anomalous_texts:
        path = item.get('path', [])
        text = item.get('text', 'N/A')
        network_info = item.get('network_info', [])

        if isinstance(path, dict):
            sorted_keys = sorted(map(int, path.keys()))
            nodes = [str(path[str(k)]).strip('"') for k in sorted_keys]
        elif isinstance(path, list):
            nodes = [str(node).strip('"') for node in path]
        else:
            print(f"Unexpected path format: {path}")
            continue

        network_info = network_info if isinstance(network_info, list) else [{}] * len(nodes)
        if len(network_info) < len(nodes):
            network_info.extend([{}] * (len(nodes) - len(network_info)))

        for i, node in enumerate(nodes):
            if node not in G:
                G.add_node(node, cmdLine=text, network_info=network_info[i])
            if i > 0:
                src, dst = nodes[i - 1], node
                if not G.has_edge(src, dst):
                    G.add_edge(src, dst, weight=1)

    subgraphs = list(nx.weakly_connected_components(G)) if G.is_directed() else list(nx.connected_components(G))
    subgraph_weight_list = []

    for idx, subgraph_nodes in enumerate(subgraphs, 1):
        subG = G.subgraph(subgraph_nodes)
        total_weight = sum(data.get('weight', 0) for _, _, data in subG.edges(data=True))

        subgraph_info = {
            "Subgraph": idx,
            "TotalWeight": total_weight,
            "Edges": [(edge[0], edge[1], subG.edges[edge].get('weight', 0)) for edge in subG.edges()],
            "Nodes": []
        }

        for node in subgraph_nodes:
            cmd = subG.nodes[node].get('cmdLine', 'N/A')
            net_info = subG.nodes[node].get('network_info', {})
            subgraph_info["Nodes"].append({
                "Node": node,
                "CmdLine": cmd,
                "NetworkInfo": net_info
            })

        subgraph_weight_list.append(subgraph_info)

    subgraph_weight_list.sort(key=lambda x: x["TotalWeight"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subgraph_weight_list, f, indent=4)

    return G

# Main execution pipeline
def run(G, sorted_shortest_paths):
    '''
    Main function to run the full SimHash-based anomaly detection pipeline
    and generate anomaly subgraphs.
    '''
    best_vectors, texts = SimHash_Recommender(G, sorted_shortest_paths)
    anomaly_indices = hard_voting_anomaly_detection(best_vectors, contamination=0.5)

    anomalous_texts = [
        {
            'text': texts[i],
            'path_length': sorted_shortest_paths[i][1][0],
            'path': sorted_shortest_paths[i][1][1],
            'community_score': sorted_shortest_paths[i][1][2],
            'network_info': [G.nodes[node].get('network_info', {}) for node in sorted_shortest_paths[i][1][1]]
        } for i in anomaly_indices
    ]

    for n_communities in [3, 5, 7, 9, 11]:
        min_score = min(item['community_score'] for item in anomalous_texts)
        max_score = max(item['community_score'] for item in anomalous_texts)
        interval = (max_score - min_score) / n_communities
        for item in anomalous_texts:
            community_label = int((item['community_score'] - min_score) / interval)
            community_label = min(community_label, n_communities - 1)
            item[f'community_{n_communities}'] = community_label

    print_anomalous_graph(anomalous_texts)
    return anomalous_texts