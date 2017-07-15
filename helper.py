import os
from collections import Counter
import random
import numpy as np


def load_books():
    corpus = ''
    books_list = os.listdir('./final_data')

    for book in books_list:
        with open('./final_data/'+book) as f:
            corpus += f.read()

    return corpus


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    word_counts = Counter(words)
    corpus = [word for word in words if word_counts[word] > 5]

    return corpus


def create_dict(corpus):
    word_counts = Counter(corpus)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def sub_sampling(encoded_corpus):
    threshold = 1e-5
    word_counts = Counter(encoded_corpus)
    total_count = len(encoded_corpus)
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    sampled_encoded_corpus = [word for word in encoded_corpus if random.random() < (1 - p_drop[word])]
    return sampled_encoded_corpus


def get_target(words, idx, window_size=5):
    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])

    return list(target_words)


def get_batches(words, batch_size, window_size=5):
    n_batches = len(words) // batch_size

    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y
