import tensorflow as tf

import os
import sys
import glob
from utils import _get_output_filename,int64_feature,bytes_feature, encode_labels, load_image


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

    labels_encord,lengths = encode_labels(labels)

    image_format = b'JPEG'
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, filename in enumerate(imgLists):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(imgLists)))
            sys.stdout.flush()
            #image_data = tf.gfile.FastGFile(filename, 'rb').read()
            image_data = load_image(filename)
            #with tf.gfile.GFile(filename, 'rb') as fid:
            #    image_data = fid.read()
            # with tf.Session() as sess:
            #     image = tf.image.decode_jpeg(image_data)
            #     image = sess.run(image)
            #     print(image.shape)#(32, 100, 3)


            example = tf.train.Example(features=tf.train.Features(feature={"label/value": int64_feature(labels_encord[i]),
                                                                            "label/length": int64_feature(lengths[i]),
                                                                           "image/encoded": bytes_feature(image_data),
                                                                           'image/format': bytes_feature(image_format)}))
            tfrecord_writer.write(example.SerializeToString())
    print('\nFinished converting the dataset!')

if __name__ == '__main__':
    train_txt = "train_list_dashes.txt"
    valid_txt = "valid_list_dashes.txt"

    train_imgs, train_labels = get_lists(train_txt)
    valid_imgs, valid_labels = get_lists(valid_txt)

    output_dir = "data"
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)
    img_to_tfrecord(train_imgs,train_labels,output_dir,"train")
    img_to_tfrecord(valid_imgs,valid_labels,output_dir,"valid")