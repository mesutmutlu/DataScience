from __future__ import division, print_function, unicode_literals
import tensorflow as tf
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tensorflow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def standard_ts_call(f):

    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print(result)

    sess.close()

def block_ts_call(f):
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
        print(result)

def block_gvi_ts_call(f):
    init = tf.global_variables_initializer()
      # prepare an init node
    with tf.Session() as sess:
        init.run()  # actually initialize all the variables
        result = f.eval()
        print(result)

def int_gvi_ts_call(f):
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print(result)
    sess.close()

if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # print(tf.__version__)

    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x * x * y + y + 2
    # print(f, type(f))
    # standard_ts_call(f)
    # block_ts_call(f)
    #
    # block_gvi_ts_call(f)
    # int_gvi_ts_call(f)

    x1 = tf.Variable(1)
    print(x1.graph is tf.get_default_graph())

    graph = tf.Graph()

    with graph.as_default():
        x2 = tf.Variable(2)
        print(x2.graph is tf.get_default_graph())
    print(x2.graph is graph)

    print(x2.graph is tf.get_default_graph())

    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3
    with tf.Session() as sess:
        print(y.eval())  # 10
        print(z.eval())  # 15

    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y_val)  # 10
        print(z_val)  # 15


