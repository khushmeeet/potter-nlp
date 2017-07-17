import time
import tensorflow as tf
import numpy as np
import helper
import random


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
tf.summary.scalar('embedding_matrix', embedding)

output_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], -1, 1), name='output_w')
output_b = tf.Variable(tf.zeros(vocab_size), name='output_b')

loss = tf.nn.sampled_softmax_loss(output_w, output_b, targets, embed, n_sample, vocab_size, name='sampled_loss')
cost = tf.reduce_mean(loss, name='cost')
optimizer = tf.train.AdamOptimizer().minimize(cost)
tf.summary.scalar('cost', cost)

merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter('summaries/run_1')

# ----- testing -----
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
valid_examples = np.append(valid_examples,
                           random.sample(range(1000, 1000 + valid_window), valid_size // 2))

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# We use the cosine distance:
norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
normalized_embedding = embedding / norm
valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

saver = tf.train.Saver()

epochs = 10
batch_size = 500
window_size = 10

with tf.Session() as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = helper.get_batches(sampled_encoded_corpus, batch_size, window_size)
        start = time.time()
        for x, y in batches:

            feed = {inputs: x,
                    targets: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 500 == 0:
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1
    save_path = saver.save(sess, "checkpoints/potter2vec.ckpt")
    embed_mat = sess.run(normalized_embedding)