from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_data(data):
    sentences = sent_tokenize(data)
    tokenized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        tokenized_sentences.append(words)
    return tokenized_sentences

def create_word_frequency(sentences):
    word_frequency = Counter()
    for sentence in sentences:
        for word in sentence:
            word_frequency[word] += 1
    return word_frequency

def create_sentence_probability(sentences, word_frequency):
    sentence_probability = {}
    sentence_length = {}
    for i, sentence in enumerate(sentences):
        probability = 0
        for word in sentence:
            probability += word_frequency[word]
        sentence_probability[i] = probability / len(sentence)
        sentence_length[i] = len(sentence)
    return sentence_probability, sentence_length

def update_word_frequency(selected_sentence, word_frequency):
    for word in selected_sentence:
        word_frequency[word] -= 1
    return word_frequency

def get_summary(sentences, sentence_probability, sentence_length, length, word_frequency):
    summary = []
    summary_length = 0
    while summary_length < length:
        max_probability = -1
        selected_sentence = None
        for i, sentence in enumerate(sentences):
            probability = sentence_probability[i] / sentence_length[i]
            if probability > max_probability:
                max_probability = probability
                selected_sentence = sentence
                selected_sentence_index = i
        summary.append(selected_sentence)
        summary_length += len(selected_sentence)
        sentence_probability.pop(selected_sentence_index)
        sentence_length.pop(selected_sentence_index)
        word_frequency = update_word_frequency(selected_sentence, word_frequency)
        if len(sentence_probability) == 0:
            break
    return summary

def sumbasic_summarize(text, length=3):
    # preprocess the data
    sentences = preprocess_data(text)

    # create the word frequency distribution
    word_frequency = create_word_frequency(sentences)

    # create the sentence probability distribution
    sentence_probability, sentence_length = create_sentence_probability(sentences, word_frequency)

    # get the summary
    summary = get_summary(sentences, sentence_probability, sentence_length, length, word_frequency)

    # return the summary as a string
    return "\n".join([" ".join(sentence) for sentence in summary])
