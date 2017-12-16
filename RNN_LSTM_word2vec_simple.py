import tensorflow as tf
import re
import numpy as np
import time
from gensim.models import Word2Vec

#data = pd.read_csv("Combined_News_DJIA.csv")
#labels = np.array(data["Label"])
#reviews =data["Top1"]
w2c_model = "simple_w2v_model"
tensor_matrix = "simple_tensor.npy"
score_vector = "simple_scores.npy"
keep_prob1 = 0.8
lstm_size = 64
lstm_layers = 2
batch_size = 100
learning_rate = 0.01
epochs = 1

w2c_model = Word2Vec.load(w2c_model)
features = np.load(tensor_matrix)
old_labels = np.load(score_vector)
labels = np.zeros(len(old_labels))
for i in range(len(old_labels)):
    if old_labels[i] >=0:
        labels[i] = 1
    else:
        labels[i] =0
vocabulary = list(w2c_model.wv.vocab.keys())
embedding = np.array([w2c_model[v] for v in vocabulary])
print("Vocabulary Size: {:d}".format(len(vocabulary)))

print(len(labels))



split_frac = 0.8

split_index = int(split_frac * len(features))

train_x, val_x = features[:split_index], features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape),
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))


# Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

with tf.name_scope("Embeddings"):
    # random embedding
    # embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    # word2vec
    embedding = embedding
    embed = tf.nn.embedding_lookup(embedding, inputs_)


def lstm_cell():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)


with tf.name_scope("RNN_layers"):
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_, logits=predictions))
# ost = tf.losses.mean_squared_error(labels_, predictions)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


time1 = time.time()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    train_loss = []
    train_acc = []
    for e in range(epochs):
        state = sess.run(initial_state)

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: keep_prob1,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            train_loss.append(loss)
            print("Epoch: {}/{}".format(e, epochs),
                  "Iteration: {}".format(iteration),
                  "Train loss: {:.3f}".format(loss))

            if iteration % 50 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
                train_acc.append(np.mean(val_acc))
            iteration += 1

    test_acc = []
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

np.save("train_loss_RNN.npy", train_loss)
np.save("train_acc_RNN.npy", train_acc)
np.save("test_acc_RNN.npy", test_acc)
time2 = time.time()
time_elapsed = time2 - time1
print("Time: %f" % time_elapsed)
