from __future__ import print_function

import sys
from pathlib import Path
sys.path.append('../')
sys.path.append('./')
sys.path.append('./datasets')

import argparse
import os
import copy

import time
from PIL import Image
import tensorflow as tf
import numpy as np
from scipy import misc

from net import model

import cv2

INPUT_SIZE = "100,32,1"
BATCH_SIZE = 1
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz-_"
snapshot_dir = './tmp_32x100/'
SAVE_DIR = './output/'

def int_to_char(number,char_list):
    if(number == len(char_list)-1):
        return ""
    return char_list[number]

def sparse_tensor_to_str1(spares_tensor, alphabet):
    """
    :param spares_tensor:
    :return: a str
    """
    print(spares_tensor)
    indices= spares_tensor.indices
    values = spares_tensor.values
    dense_shape = spares_tensor.dense_shape

    number_lists = np.ones(dense_shape,dtype=values.dtype)
    str_lists = []
    strings = []
    for i,index in enumerate(indices):
        number_lists[index[0],index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([int_to_char(val, alphabet) for val in number_list])
    for str_list in str_lists:
        strings.append("".join(str_list))
    return strings

def sparse_tensor_to_str(indices, values, dense_shape, alphabet):
    """
    :param spares_tensor:
    :return: a str
    """
    #print(spares_tensor)
    #indices= spares_tensor.indices
    #values = spares_tensor.values
    #dense_shape = spares_tensor.dense_shape

    number_lists = np.ones(dense_shape,dtype=values.dtype)
    str_lists = []
    strings = []
    for i,index in enumerate(indices):
        number_lists[index[0],index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([int_to_char(val,alphabet) for val in number_list])
    for str_list in str_lists:
        strings.append("".join(str_list))
    return strings

def GetAllFilesListRecusive(path, extensions):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
             # linux tricks with .directory that still is file
            if not 'directory' in name and sum([ext in name for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all

def calculate_perfomance(sess, input, raw_output, shape, runs = 1000, batch_size = 1):

    start = time.time()

    print('Calculating inference time...\n')
    # To exclude numpy generating time
    N = 10
    for i in range(0, N):
        img = np.random.random((batch_size, shape[0], shape[1], shape[2]))
    stop = time.time()
    
    # warm up
    sess.run(raw_output, feed_dict = {input : img})

    time_for_generate = (stop - start) / N

    start = time.time()
    for i in range(runs):
        img = np.random.random((batch_size, shape[0], shape[1], shape[2]))
        sess.run(raw_output, feed_dict = {input : img})

    stop = time.time()

    inf_time = ((stop - start) / float(runs)) - time_for_generate

    print('Average inference time: {}'.format(inf_time))


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.",
                        required=True)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--snapshots-dir", type=str, default=snapshot_dir,
                        help="Path to checkpoints.")
    parser.add_argument("--alphabet", type=str, default=ALPHABET,
                        help="model's alphabet if checkpoints is used")
    parser.add_argument("--pb-file", type=str, default='',
                        help="Path to to pb file, alternative for checkpoint. If set, checkpoints will be ignored")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Size of batch for time measure")
    parser.add_argument("--measure-time", action="store_true", default=False,
                        help="Evaluate only model inference time")
    parser.add_argument("--runs", type=int, default=100,
                        help="Repeats for time measure. More runs - longer testing - more precise results")


    return parser.parse_args()


def load_img(img_path, crop_size):

    filename = img_path.split('/')[-1]
    img = cv2.imread(img_path,0)
    img = cv2.resize(img,(crop_size[0],crop_size[1]))
    if crop_size[2]==1:
        im_arr = np.reshape(img, (img.shape[0], img.shape[1], 1))
    else:
        im_arr = img
    print('input image shape: ', img.shape)
    
    return im_arr, filename


def load_from_checkpoint(shape, checkpoint_dir):
    #width_input = tf.placeholder(tf.int32, shape=())
    img_input = tf.placeholder(tf.float32, shape=(None, shape[1], shape[0], shape[2]))
    input_batch = tf.placeholder(tf.int32, shape=())
    img_input = img_input * (1. / 255) - 0.5
    #img_4d = tf.expand_dims(img_input, 0)

    # define the crnn net
    SEQ_LEN = int(shape[0]/4+1)
    crnn_params = model.CRNNNet.default_params._replace(seq_length = SEQ_LEN)._replace(imgH = shape[1])  # ,seq_length=int(width/4+1)
    crnn = model.CRNNNet(crnn_params)
    logits, inputs, seq_len, W, b = crnn.net(img_input, input_batch)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    val_predict = tf.cast(decoded[0], tf.int32)

    saver = tf.train.Saver()

    sess = tf.Session()
    dir = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, dir)
    sess.run(tf.local_variables_initializer())
    print("Model restore!")
    return sess, val_predict, img_input, input_batch

def load_from_pb(path):
    graph = tf.Graph()
    with graph.as_default():
        seg_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            seg_graph_def.ParseFromString(serialized_graph)

            tf.import_graph_def(seg_graph_def, name = '')

            x = graph.get_tensor_by_name('input_tensor:0')
            input_batch = graph.get_tensor_by_name('input_batch_size:0')

            pred_ind = graph.get_tensor_by_name('sparse_tensor_indices:0')
            pred_val = graph.get_tensor_by_name('sparse_tensor_values:0')
            pred_shape = graph.get_tensor_by_name('sparse_tensor_shape:0')

            print("GRAPH OUTPUT: ", pred_ind,pred_val,pred_shape)
            shape_tensor = graph.get_tensor_by_name('input_plate_size:0')
            alphabet_tensor = graph.get_tensor_by_name('alphabet:0')
            
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.do_common_subexpression_elimination = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            config.allow_soft_placement = True
            config.log_device_placement = False

            sess = tf.Session(graph = graph, config = config)
            shape, alphabet = sess.run([shape_tensor,alphabet_tensor])
            alphabet = alphabet.decode("utf-8")
            print("Alphabet: ", alphabet)

    return sess, pred_ind, pred_val, pred_shape, x, input_batch, shape, alphabet

def main():
    args = get_arguments()
    
    if args.img_path[-4] != '.':
        files = GetAllFilesListRecusive(args.img_path, ['.jpg', '.jpeg', '.png', '.JPG'])
    else:
        files = [args.img_path]


    if args.pb_file == '':
        shape = INPUT_SIZE.split(',')
        shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        sess, pred, x, input_batch = load_from_checkpoint(shape, args.snapshots_dir)
        alphabet = args.alphabet
    else:
        sess, pred_ind, pred_val, pred_shape, x,input_batch, shape, alphabet = load_from_pb(args.pb_file)
    
    w,h,c = shape

    if args.measure_time:
        calculate_perfomance(sess, x, pred, shape, args.runs, args.batch_size)
        quit()

    for path in files:

        img, filename = load_img(path, (w,h,c))   
        orig_img = copy.deepcopy(img)

        exanded_img = np.expand_dims(img, axis = 0)

        t = time.time()
        if args.pb_file != '':
            ind,val,sh = sess.run([pred_ind, pred_val, pred_shape], feed_dict = {x: exanded_img, input_batch:1})
            result = sparse_tensor_to_str(ind,val,sh,alphabet)[0]
        else:
            preds = sess.run(pred, feed_dict = {x: exanded_img, input_batch:1})
            result = sparse_tensor_to_str1(preds,alphabet)[0]
        print('time: ', time.time() - t)

        
        print("Result: ", result)
        cv2.putText(img,result,(0,h),cv2.FONT_HERSHEY_SIMPLEX,0.25,(255,255,255),1)
                
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)        

        cv2.imwrite(args.save_dir + filename.split(".")[0] + "_out." + filename.split(".")[-1], img)
        cv2.imshow("result",img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
