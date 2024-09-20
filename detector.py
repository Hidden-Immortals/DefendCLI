import networkx as nx
import numpy as np
import re
from joblib import Parallel, delayed
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from itertools import combinations
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Preprocessing function for text
def preprocess_text(text):
    """
    Converts text to lowercase and removes non-alphanumeric characters.
    Splits the text into words.
    """
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.split()

# Function to compute SimHash
def simhash(vec, bits=64):
    """
    Computes SimHash for a given vector.
    """
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

# Function to compute Hamming distance
def hamming_distance(hash1, hash2):
    """
    Computes the Hamming distance between two SimHash values.
    """
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# Function to train a Word2Vec model
def train_word2vec(texts, vector_size=60, window=5, min_count=1, workers=4):
    """
    Trains a Word2Vec model on the given texts.
    """
    processed_texts = [preprocess_text(text) for text in texts]
    model = Word2Vec(sentences=processed_texts, vector_size=vector_size, window=window, min_count=min_count,
                     workers=workers)
    return model

# Function to train a FastText model
def train_fasttext(texts, vector_size=60, window=5, min_count=1, workers=4):
    """
    Trains a FastText model on the given texts.
    """
    processed_texts = [preprocess_text(text) for text in texts]
    model = FastText(sentences=processed_texts, vector_size=vector_size, window=window, min_count=min_count,
                     workers=workers)
    return model

# Function to train a Doc2Vec model
def train_doc2vec(texts, vector_size=60, epochs=20, workers=4):
    """
    Trains a Doc2Vec model on the given texts.
    """
    tagged_data = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(texts)]
    model = Doc2Vec(vector_size=vector_size, epochs=epochs, workers=workers)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Function to convert text to vector using a given model
def text_to_vector(text, model):
    """
    Converts text to a vector using the provided model.
    """
    words = preprocess_text(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def text_to_vector_doc2vec(text, model):
    """
    Converts text to a vector using a Doc2Vec model.
    """
    return model.infer_vector(preprocess_text(text))

# Function to calculate SimHash for texts
def calculate_simhash_for_texts(texts, model, doc2vec_model=False):
    """
    Calculates SimHash values for the given texts using the provided model.
    """
    if doc2vec_model:
        vectors = [text_to_vector_doc2vec(text, model) for text in texts]
    else:
        vectors = [text_to_vector(text, model) for text in texts]
    simhashes = [simhash(vector) for vector in vectors]
    return simhashes, vectors

# Function to evaluate SimHash distances
def evaluate_simhashes(simhashes):
    """
    Evaluates SimHash distances by calculating the mean Hamming distance between all pairs of SimHash values.
    """
    distances = [hamming_distance(hash1, hash2) for hash1, hash2 in combinations(simhashes, 2)]
    return np.mean(distances) if distances else 0

# Hard voting anomaly detection function
def hard_voting_anomaly_detection(vectors, contamination=0.5):
    """
    Performs anomaly detection using an ensemble of different algorithms and combines their results using hard voting.
    """
    models = {
        'IsolationForest': IsolationForest(
            n_estimators=16,  # Increase the number of trees
            max_samples=0.5,  # Decrease max samples
            contamination=contamination,
            max_features=1.0,
            bootstrap=True,
            random_state=42
        ),
        'LocalOutlierFactor': LocalOutlierFactor(
            n_neighbors=16,  # Decrease number of neighbors
            contamination=contamination,
            algorithm='auto',
            leaf_size=30,  # Increase leaf size
            metric='minkowski',
            p=2,
            n_jobs=-1
        ),
        'OneClassSVM': OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='scale',
            tol=1e-4,  # Increase tolerance
            shrinking=True,
            cache_size=200,
            max_iter=-1
        ),
        'EllipticEnvelope': EllipticEnvelope(
            contamination=contamination,
            support_fraction=1,  # Increase support fraction
            random_state=42
        ),
        'KMeans': KMeans(
            n_clusters=16,  # Increase number of clusters
            random_state=42,
            n_init=20,  # Increase number of initializations
            max_iter=300,  # Increase maximum iterations
            tol=1e-4,
            algorithm='lloyd'
        ),
        'GaussianMixture': GaussianMixture(
            n_components=16,  # Increase number of components
            covariance_type='full',
            tol=1e-3,  # Increase tolerance
            reg_covar=1e-6,
            max_iter=200,  # Increase maximum iterations
            n_init=5,  # Increase number of initializations
            init_params='kmeans',
            random_state=42
        )
    }

    def fit_predict_model(name, model, vectors):
        """
        Fits the model to the vectors and predicts anomalies.
        """
        try:
            if name == 'LocalOutlierFactor':
                predictions = model.fit_predict(vectors)
            else:
                model.fit(vectors)
                predictions = model.predict(vectors)
            return name, predictions
        except ValueError as e:
            print(f"Error with model {name}: {e}")
            return name, np.ones(len(vectors))  # Return normal prediction if error occurs

    results = Parallel(n_jobs=-1)(delayed(fit_predict_model)(name, model, vectors) for name, model in models.items())

    predictions = {}
    for name, pred in results:
        if name == 'KMeans' or name == 'GaussianMixture':
            pred = np.where(pred == 0, 1, -1)
        if name == 'LocalOutlierFactor':
            pred = np.where(pred == 1, 1, -1)
        predictions[name] = pred

    combined_anomalies = sum((pred == -1).astype(int) for pred in predictions.values())
    final_anomalies = np.where(combined_anomalies >= 4)[0]  # Keep the threshold for anomalies at 4

    return final_anomalies

def convert_cmdLine_to_str(cmdLine):
    """
    Converts a set of command lines to a single string.
    """
    if isinstance(cmdLine, set):
        return ' '.join(cmdLine)
    return cmdLine

# SimHash recommender system function
def SimHash_Recommender(G, sorted_shortest_paths):
    """
    Recommends potential anomalies in the graph using SimHash and different embedding models.
    """
    # E3-Dataset
    # texts = [
    #    ' '.join([G.nodes[node].get('cmdLine', 'N/A') for node in path])
    #    for _, (_, path, _) in sorted_shortest_paths
    # ]
    
    # Ours
    texts = [
        '\n'.join([convert_cmdLine_to_str(G.nodes[node].get('cmdLine', 'N/A')) for node in path])
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
        print("Word2Vec has the maxium score.")
        best_vectors = word2vec_vectors
    elif max_score == doc2vec_score:
        print("Doc2Vec has the maxium score.")
        best_vectors = doc2vec_vectors
    else:
        print("FastText has the maxium score.")
        best_vectors = fasttext_vectors
    return best_vectors, texts

# Main function
def run(G, sorted_shortest_paths):
    """
    Main function to detect anomalies in the graph using SimHash and hard voting anomaly detection.
    """
    best_vectors, texts = SimHash_Recommender(G, sorted_shortest_paths)
    anomaly_indices = hard_voting_anomaly_detection(best_vectors, contamination=0.5)  # Keep contamination rate at 0.5
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
    return anomalous_texts
