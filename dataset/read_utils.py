import tensorflow as tf
from dataset.utils import char_to_int
import numpy as np


def prepare_image(img, crop_size):
    h, w = crop_size
    initial_width = tf.shape(img)[1]
    initial_height = tf.shape(img)[0]    

    new_h = tf.to_int32(h)
    ratio = tf.constant(h, dtype=tf.float32)/ tf.to_float(initial_height)
    new_w = ratio * tf.to_float(initial_width)
    new_w = tf.to_int32(new_w)
    
    img = tf.image.resize_images(img, [new_h,new_w])
    img = tf.image.pad_to_bounding_box(img, 0, 0, h, tf.maximum(w,new_w))

    img = tf.random_crop(img, [h,w,3])
    img.set_shape([h, w, 3])

    img = tf.image.rgb_to_grayscale(img)
    return img

#tfrecord must be created with create_dataset.py
def read_and_decode(filename, num_epochs, crop_size):  # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)  # return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature((), tf.string, default_value=''),
                                           'label/value': tf.VarLenFeature(tf.int64),
                                           'label/length': tf.FixedLenFeature([1], tf.int64)
                                       })  # return image and label

    
    impath = features['image']

    img_contents = tf.read_file(tf.convert_to_tensor(impath, dtype=tf.string))        
    img = tf.image.decode_jpeg(img_contents, channels=3)

    #img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    #img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    
    img = tf.cast(img, dtype=tf.float32)
    # Extract mean.
    #image = img - img_mean
    image = img * (1. / 255) - 0.5
    
    img = image

    #crop_size = [32,100]
    img = prepare_image(img, crop_size)

    label = features['label/value']  # throw label tensor
    label = tf.cast(label, tf.int32)
    length = features["label/length"]
    #length = tf.cast(length, tf.int32)

    return img, label, impath, length
  

def inputs(batch_size, num_epochs, filename, crop_size = [32,100]):
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        # Even when reading in multiple threads, share the filename
        # queue.
        img, label, path, lengt = read_and_decode(filename, num_epochs, crop_size)
        

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        
        sh_images, sh_labels, sh_path = tf.train.shuffle_batch(
            [img, label, path], batch_size=batch_size, num_threads=2,
            capacity=50000,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=500)
        '''
        sh_images, sh_labels, sh_path = tf.train.batch([img, label, path], batch_size=batch_size)
        '''
        return sh_images, sh_labels, sh_path

def preprocess_for_train(image,label ,scope='crnn_preprocessing_train'):
    """Preprocesses the given image for training.
    """
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)# convert image as a tf.float32 tensor
            image_s = tf.expand_dims(image, 0)
            tf.summary.image("image",image_s)

        image = tf.image.rgb_to_grayscale(image)
        tf.summary.image("gray",image)
        return image, label