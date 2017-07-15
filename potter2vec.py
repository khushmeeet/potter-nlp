import time
import tensorflow as tf
import numpy as np
import helper


corpus = helper.load_books()
corpus = helper.preprocess(corpus)
vocab_to_int, int_to_vocab = helper.create_dict(corpus)
encoded_corpus = [vocab_to_int[word] for word in corpus]
sampled_encoded_corpus = helper.sub_sampling(encoded_corpus)

vocab_size = len(vocab_to_int)
embedding_size = 300
n_sample = 100

inputs = tf.placeholder(tf.int32, [None], name='inputs')
targets = tf.placeholder(tf.int32, [None, None], name='targets')

embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size], -1, 1), name='embedding_matrix')
embed = tf.nn.embedding_lookup(embedding, inputs, name='embedding_lookup')

output_w = tf.Variable(tf.truncated_normal([embedding_size, vocab_size], -1, 1), name='output_w')
output_b = tf.Variable(tf.zeros(vocab_size), name='output_b')

loss = tf.nn.sampled_softmax_loss(output_w, output_b, targets, embed, n_sample, vocab_size, name='sampled_loss')
cost = tf.reduce_mean(loss, name='cost')
optimizer = tf.train.AdamOptimizer().minimize(cost)
