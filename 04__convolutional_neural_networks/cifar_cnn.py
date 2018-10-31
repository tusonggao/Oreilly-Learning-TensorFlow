import pickle
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer


DATA_PATH = "C:/github_base/Oreilly-Learning-TensorFlow/data_dir/cifar_data"
BATCH_SIZE = 50
#BATCH_SIZE = 70
#STEPS = 500000
#STEPS = 300000
STEPS = 2000

def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict

def display_cifar_sorted(images, labels, size):
    # print('labels shape is', labels.shape)
    # print('np.argwhere(labels==2) shape ', np.argwhere(labels==2).shape)
    # print('shape2 is ', np.random.choice(np.argwhere(labels == 2).ravel(), 10))
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack(images[np.random.choice(np.argwhere(labels==i).ravel(), 10)])
                    for i in range(size)])
    # print('im shape ', im.shape)
    plt.imshow(im)
    plt.show()

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()

def plot_accuracy(iter_list, test_accuracy_list, train_accuracy_list=None):
    import matplotlib.pyplot as plt
    plt.plot(iter_list, test_accuracy_list, color='red', label='test accuracy')
    if train_accuracy_list:
        plt.plot(iter_list, train_accuracy_list, color='blue', label='train accuracy')
        plt.legend()
    plt.xlabel('iter num')
    plt.ylabel('accuracy')
    plt.show()


class CifarLoader(object):
    """
    Load and mange the CIFAR dataset.
    (for any practical use there is no reason not to use the built-in dataset handler instead)
    """
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None
        self.raw_labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
            .astype(float) / 255
        self.raw_labels = np.hstack([d["labels"] for d in data])
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], \
               self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n, batch_size)
        return self.images[ix], self.labels[ix]


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)])\
            .load()
        self.test = CifarLoader(["test_batch"]).load()


def run_simple_net():
    print('get into run_simple_net')
    cifar = CifarDataManager()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    conv1 = conv_layer(x, shape=[5, 5, 3, 32])
    conv1_pool = max_pool_2x2(conv1)

    print('get into run_simple_net 111')

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)

    # conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
    conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 100])
    conv3_pool = max_pool_2x2(conv3)

    conv4 = conv_layer(conv3_pool, shape=[5, 5, 100, 256])
    conv4_pool = max_pool_2x2(conv4)
    print('conv4_pool.get_shape() is ', conv4_pool.get_shape())
    #conv4_flat = tf.reshape(conv4_pool, [-1, 4 * 4 * 256])
    conv4_flat = tf.reshape(conv4_pool, [-1, 2 * 2 * 256])
    conv4_drop = tf.nn.dropout(conv4_flat, keep_prob=keep_prob)

    print('get into run_simple_net 222')

    full_1 = tf.nn.relu(full_layer(conv4_drop, 512))
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                                           labels=y_))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test(sess):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                       for i in range(10)])
        #print("test accuracy: {:.4}%".format(acc * 100))
        return acc

    print('get into run_simple_net 333')
    iter_list, test_accuracy_list, train_accuracy_list = [], [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            #batch = cifar.train.next_batch(BATCH_SIZE)
            batch = cifar.train.random_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 500 == 0:
                iter_list.append(i)
                train_acc = np.mean([sess.run(accuracy,
                            feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                            for i in range(10)])
                train_accuracy_list.append(train_acc)
                test_acc = test(sess)
                test_accuracy_list.append(test_acc)

                print('step i:{0:6} train_acc: {1:.6f} test_acc: {2:.6f}'.format(
                      i, train_acc, test_acc))

        test(sess)
        plot_accuracy(iter_list, test_accuracy_list, train_accuracy_list)


def build_second_net():
    cifar = CifarDataManager()
    print('in build_second_net 111')

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    C1, C2, C3 = 32, 64, 128
    F1 = 600

    conv1_1 = conv_layer(x, shape=[3, 3, 3, C1])
    conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])
    conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])
    conv1_pool = max_pool_2x2(conv1_3)
    conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

    print('in build_second_net 222')

    conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])
    conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])
    conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])
    conv2_pool = max_pool_2x2(conv2_3)
    conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

    print('in build_second_net 333')

    conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])
    conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])
    conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])
    conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    conv3_flat = tf.reshape(conv3_pool, [-1, C3])
    conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

    full1 = tf.nn.relu(full_layer(conv3_drop, F1))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                                           labels=y_))
    train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)  # noqa

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # noqa

    # Plug this into the test procedure as above to continue...
    def test(sess):
        X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
        Y = cifar.test.labels.reshape(10, 1000, 10)
        acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0})
                       for i in range(10)])
        #print("Accuracy: {:.4}%".format(acc * 100))
        return acc

    print('get into run_simple_net 333')
    iter_list, test_accuracy_list, train_accuracy_list = [], [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            #print('i is ', i)
            #batch = cifar.train.next_batch(BATCH_SIZE)
            batch = cifar.train.random_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 500 == 0 or i==STEPS-1:
                iter_list.append(i)
                train_acc = sess.run(accuracy, feed_dict={x: batch[0],
                                                          y_: batch[1],
                                                          keep_prob: 0.5})
                train_accuracy_list.append(train_acc)
                test_acc = test(sess)
                test_accuracy_list.append(test_acc)
                print(
                    'step i:{0:7} train_acc: {1:.6f} test_acc: {2:.6f}'.format(
                        i, train_acc, test_acc))

        test(sess)
        plot_accuracy(iter_list, test_accuracy_list, train_accuracy_list)


def create_cifar_image():
    d = CifarDataManager()
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test labels: {}".format(len(d.test.labels)))
    images = d.train.images
    labels = d.train.labels
    raw_labels = d.train.raw_labels
    #display_cifar(images, 10)
    #display_cifar_sorted(images, raw_labels, 10)


if __name__ == "__main__":
    start_t = time.time()
    create_cifar_image()
    #run_simple_net()
    build_second_net()
    print('total cost time ', time.time()-start_t)
