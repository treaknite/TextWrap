import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np

def preprocess_data(data):
    sentences = sent_tokenize(data)
    tokenized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        tokenized_sentences.append(words)
    return tokenized_sentences

def create_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = 1 - cosine_distance(sentences[i], sentences[j])
    return similarity_matrix

def calculate_lexrank(similarity_matrix, damping_factor=0.85, max_iterations=100, epsilon=1e-4):
    nx_graph = nltk.to_networkx_graph(similarity_matrix)
    scores = nx.pagerank(nx_graph, alpha=damping_factor, max_iter=max_iterations, tol=epsilon)
    return scores

def get_summary(sentences, scores, n=3):
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    summary = []
    for i in range(n):
        summary.append(ranked_sentences[i][1])
    return summary

def lexrank_summarize(text, n=3):
    # preprocess the data
    sentences = preprocess_data(text)

    # create the similarity matrix
    similarity_matrix = create_similarity_matrix(sentences)

    # calculate the LexRank scores
    scores = calculate_lexrank(similarity_matrix)

    # get the summary
    summary = get_summary(sentences, scores, n=n)

    # return the summary as a string
    return "\n".join([" ".join(sentence) for sentence in summary])
