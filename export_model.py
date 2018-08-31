r"""Saves out a GraphDef containing the architecture of the model."""

from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.platform import gfile

from net import model

input_size = (100,32,1)
batch = 1
SEQ_LEN = int(input_size[0]/4+1)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz-_"
#tf.app.flags.DEFINE_integer(
#    'image_size', None,
#    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        shape = input_size
        shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        img_input = tf.placeholder(name = 'input_tensor', dtype = tf.float32, shape=(None, input_size[1], input_size[0], int(shape[2])))
        img_input = img_input * (1. / 255) - 0.5
        batch_input = tf.placeholder(name = 'input_batch_size', dtype =tf.int32, shape=())
        #img_4d = tf.expand_dims(img_input, 0)

        print(img_input)
        # Create network.
        crnn_params = model.CRNNNet.default_params._replace(imgH = input_size[1])._replace(seq_length = SEQ_LEN)  # ,seq_length=int(width/4+1)
        crnn = model.CRNNNet(crnn_params)
        
        logits, inputs, seq_len, W, b = crnn.net(img_input, batch_input)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        
        indices = tf.cast(decoded[0].indices, tf.int32, name = 'sparse_tensor_indices')
        values = tf.cast(decoded[0].values, tf.int32, name = 'sparse_tensor_values')
        dense_shape = tf.cast(decoded[0].dense_shape, tf.int32, name = 'sparse_tensor_shape')

        print(indices,values,dense_shape)

        tf.constant(shape, name = 'input_plate_size')
        tf.constant(alphabet, dtype = tf.string, name = 'alphabet')
        print(alphabet)
        tf.constant(["sparse_tensor"], name = "output_names")
        tf.constant(['input_tensor'], name = 'input_name')

        graph_def = graph.as_graph_def()
        with gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
            print('Successfull written to', FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()