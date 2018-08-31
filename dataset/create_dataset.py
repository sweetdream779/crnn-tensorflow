import tensorflow as tf

import os
import sys
import glob
from utils import _get_output_filename,int64_feature,bytes_feature, encode_labels, load_image

#last symbol for blank output
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz_"

def get_lists(filepath):
    images = []
    labels = []
    print("\nGetting lists...")
    count_lines = len(open(filepath, "r").readlines())
    num = 0
    with open(filepath, "r") as f:
        for line in f:
            num+=1
            sys.stdout.write('\r>> Read sample %d/%d' % (num, count_lines))
            sys.stdout.flush()
            
            line = line.split("\n")[0]
            image_path, annot_path = line.split(" ")
            images.append(image_path)
            
            ann = open(annot_path, "r")
            first_line = ann.readline()
            first_line = first_line.split("\n")[0].lower()

            #######
            first_line = first_line.replace("-","")
            #######
            labels.append(first_line)

            ann.close()
    print("\nGot images and labels")
    combined = list(zip(labels, images))
    combined.sort(key = lambda s: len(s[0]))
    labels[:], images[:] = zip(*combined)

    return images, labels


def img_to_tfrecord(train_imgs,train_labels,output_dir,name):
    """

    :param image_dir: image_dir just like "data/Challenge2_Training_Task12_Images/*.jpg"
    :param text_dir: label file dir
    :param text_name: label file name
    :param output_dir: output dir
    :param name: output file name
    :return: NULL
    """

    tf_filename = _get_output_filename(output_dir, name)

    imgLists = train_imgs  # return a list
    labels = train_labels

    labels_encord,lengths = encode_labels(labels,alphabet)

    image_format = b'JPEG'
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, filename in enumerate(imgLists):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(imgLists)))
            sys.stdout.flush()
            example = tf.train.Example(features=tf.train.Features(feature={"label/value": int64_feature(labels_encord[i]),
                                                                            "label/length": int64_feature(lengths[i]),
                                                                           "image": bytes_feature(filename)}))
            tfrecord_writer.write(example.SerializeToString())
    print('\nFinished converting the dataset!')

if __name__ == '__main__':
    train_txt = "train_list_dashes2.txt"
    valid_txt = "valid_list_dashes2.txt"

    train_imgs, train_labels = get_lists(train_txt)
    valid_imgs, valid_labels = get_lists(valid_txt)

    output_dir = "data"
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)
    img_to_tfrecord(train_imgs,train_labels,output_dir,"train4")
    img_to_tfrecord(valid_imgs,valid_labels,output_dir,"valid4")