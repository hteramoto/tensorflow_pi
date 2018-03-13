#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import argparse
import os.path as path


def build_gregory_leibniz_series():
    """
    Calculate Pi using Gregory–Leibniz series
            oo
            __        n
            \     (-1)
    pi = 4  /   -------
            --   2n + 1
            n=0
    """
    with tf.name_scope('gls'):
        n = tf.Variable(0, name='n')
        pi = tf.Variable(0, name='pi', dtype=tf.float64)
        gregory_leibniz = tf.multiply(np.float64(4), tf.divide(tf.pow(-1, n), tf.add(tf.multiply(2, n), 1)))

        update_pi = tf.assign_add(pi, gregory_leibniz)

        with tf.control_dependencies([update_pi]):
            update_n = tf.assign_add(n, 1)

        tf.summary.scalar('gls_pi', update_pi)
        merged = tf.summary.merge_all()

        return 'Gregory Leibniz series', [update_pi, update_n, merged]


def build_bbp():
    """
    Calculate Pi using BBP digit extraction
            oo
            __
            \     1     /    4            2            1             1       \
    pi =    /   ----   |  --------  -  --------  -  --------  -  ---------   |
            --   16^k   \  8k + 1       8k + 4       8k + 5        8k + 6    /
            k=0
    """
    with tf.name_scope('bbp'):
        k = tf.Variable(0, name='k', dtype=tf.float64)
        pi = tf.Variable(0, name='pi', dtype=tf.float64)
        first_part = tf.divide(np.float64(4), tf.add(tf.multiply(np.float64(8), k), 1))
        second_part = tf.divide(np.float64(2), tf.add(tf.multiply(np.float64(8), k), 4))
        third_part = tf.divide(np.float64(1), tf.add(tf.multiply(np.float64(8), k), 5))
        fourth_part = tf.divide(np.float64(1), tf.add(tf.multiply(np.float64(8), k), 6))
        subtract_result = tf.subtract(tf.subtract(tf.subtract(first_part, second_part), third_part), fourth_part)
        bailey = tf.multiply(tf.pow(tf.reciprocal(np.float64(16)), k), subtract_result)

        update_pi = tf.assign_add(pi, bailey)
        with tf.control_dependencies([update_pi]):
            update_k = tf.assign_add(k, 1)
        tf.summary.scalar('bbp_pi', update_pi)
        merged = tf.summary.merge_all()

        return 'BBP digit extraction', [update_pi, update_k, merged]


def run_pi_calculation(description, graphs, iteration=15000, print_every=1000, debug=False, log_path=None):
    with tf.Session() as sess:
        train_writer = None
        if log_path:
            train_writer = tf.summary.FileWriter(path.join(log_path, description.split(' ')[0]), sess.graph)
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        print('Running pi approximation algorithm {}:'.format(description))
        for i in range(1, iteration+1):
            result, _, summary = sess.run(graphs)
            if train_writer:
                train_writer.add_summary(summary, i)
            if i % print_every == 0:
                print('iteration {:>10}:  Pi = {}'.format(i, result))
        print()
        if train_writer:
            train_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Enables TensorFlow debugger.", action="store_true")
    parser.add_argument("-l", "--logPath", help="Saves TensorBoard logs at path.")
    args = parser.parse_args()

    gls = build_gregory_leibniz_series()
    bbp = build_bbp()

    run_pi_calculation(gls[0], gls[1], debug=args.debug, log_path=args.logPath)
    run_pi_calculation(bbp[0], bbp[1], iteration=15, print_every=1, debug=args.debug, log_path=args.logPath)
