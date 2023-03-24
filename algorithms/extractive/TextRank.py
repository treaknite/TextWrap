import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(data):
    sentences = sent_tokenize(data)
    tokenized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        tokenized_sentences.append(words)
    return tokenized_sentences

def create_word_frequency(sentences):
    word_frequency = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_frequency.keys():
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1
    return word_frequency

def create_word_embeddings(word_frequency, sentences, embedding_size=100, epochs=5, learning_rate=0.1):
    words = list(word_frequency.keys())
    word_idx = {word: i for i, word in enumerate(words)}
    vocab_size = len(words)
    W = np.random.randn(embedding_size, vocab_size)
    C = np.zeros((vocab_size, embedding_size))
    for sentence in sentences:
        for i, word in enumerate(sentence):
            word_idx_c = [word_idx[c] for c in sentence[max(i - 2, 0):i + 3]]
            C[word_idx[word]] += np.mean(W[:, word_idx_c], axis=1)
    for epoch in range(epochs):
        loss = 0
        for sentence in sentences:
            sentence_words = [word for word in sentence if word in words]
            combinations_words = list(combinations(sentence_words, 2))
            for word_i, word_j in combinations_words:
                diff = C[word_idx[word_i]] - C[word_idx[word_j]]
                dot = np.dot(W[:, word_idx[word_j]], diff)
                sigmoid = 1 / (1 + np.exp(-dot))
                grad = (1 - sigmoid) * learning_rate
                loss += grad * diff
                W[:, word_idx[word_j]] += grad * C[word_idx[word_i]]
        C *= 0
        for sentence in sentences:
            for i, word in enumerate(sentence):
                word_idx_c = [word_idx[c] for c in sentence[max(i - 2, 0):i + 3]]
                C[word_idx[word]] += np.mean(W[:, word_idx_c], axis=1)
        print("Epoch:", epoch, "Loss:", loss)
    return W

def create_sentence_similarity_matrix(sentences, word_embeddings):
    sentence_embeddings = []
    for sentence in sentences:
        embedding = np.zeros_like(word_embeddings[:, 0])
        for word in sentence:
            embedding += word_embeddings[:, list(word_frequency.keys()).index(word)]
        embedding /= len(sentence)
        sentence_embeddings.append(embedding)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity_matrix[i][j] = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[j]])[0, 0]
            similarity_matrix[j][i] = similarity_matrix[i][j]
    return similarity_matrix

def create_sentence_graph(similarity_matrix):
    graph = {}
    for i in range(len(similarity_matrix)):
        graph[i] = []
        for j in range(len(similarity_matrix)):
            if i != j and similarity_matrix[i][j] != 0:
                graph[i].append(j)
    return graph

def create_sentence_graph(similarity_matrix):
    graph = {}
    for i in range(len(similarity_matrix)):
        graph[i] = []
        for j in range(len(similarity_matrix)):
            if i != j and similarity_matrix[i][j] != 0:
                graph[i].append(j)
    return graph

def calculate_page_rank(graph, max_iterations=50, d=0.85, eps=1.0e-8):
    nodes = len(graph)
    pr = np.ones(nodes) / nodes
    out_degree = {node: len(graph[node]) for node in graph}
    for i in range(max_iterations):
        pr_new = np.ones(nodes) * (1 - d) / nodes
        for node in graph:
            if out_degree[node] == 0:
                pr_new += d * pr[node] / nodes
            else:
                for neighbor in graph[node]:
                    pr_new += d * pr[neighbor] / out_degree[neighbor]
        if np.abs(pr - pr_new).sum() < eps:
            break
        pr = pr_new
    return pr


def extract_top_sentences(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a TF-IDF matrix of the sentences
    vectorizer = TfidfVectorizer()
    sentence_matrix = vectorizer.fit_transform(sentences)

    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(sentence_matrix)

    # Create the sentence graph
    sentence_graph = create_sentence_graph(similarity_matrix)

    # Calculate the PageRank scores
    pagerank_scores = calculate_page_rank(sentence_graph)

    # Get the top sentences based on their PageRank scores
    top_sentence_indices = pagerank_scores.argsort()[-num_sentences:][::-1]
    top_sentences = [sentences[i] for i in top_sentence_indices]

    return ' '.join(top_sentences)

