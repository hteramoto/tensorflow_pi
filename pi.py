#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


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
    n = tf.Variable(0, name='n')
    pi = tf.Variable(0, name='pi', dtype=tf.float64)
    gregory_leibniz = tf.multiply(np.float64(4), tf.divide(tf.pow(-1, n), tf.add(tf.multiply(2, n), 1)))

    update_pi = tf.assign_add(pi, gregory_leibniz)

    with tf.control_dependencies([update_pi]):
        update_n = tf.assign_add(n, 1)

    return [update_pi, update_n]


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

    return [update_pi, update_k]


def run_pi_calculation(algo_name, graphs, iteration=15000, print_every=1000):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Running pi approximation algorithm {}:'.format(algo_name))
        for i in range(1, iteration+1):
            result = sess.run(graphs)
            if i % print_every == 0:
                print('iteration {:>10}:  Pi = {}'.format(i, result[0]))
        print()


if __name__ == "__main__":
    gls = build_gregory_leibniz_series()
    bbp = build_bbp()

    run_pi_calculation('Gregory Leibniz series', gls)
    run_pi_calculation('BBP digit extraction', bbp, iteration=15, print_every=1)
