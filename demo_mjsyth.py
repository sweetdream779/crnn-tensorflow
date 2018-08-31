import tensorflow as tf
from net import model
from PIL import Image
from dataset.utils import  load_label_from_img_dir, encode_label
import glob
import os
import math

import time
from dataset.utils import int_to_char

import cv2

import numpy as np

#input_size = (100,32)
#SEQ_LEN = int(input_size[0]/4 + 1)
#checkpoint_dir = './tmp/'
from train_mjsyth import SEQ_LEN,crnn_params,model_size,checkpoint_dir

input_size = (model_size[1], model_size[0])

def sparse_tensor_to_str(spares_tensor):
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
        str_lists.append([int_to_char(val) for val in number_list])
    for str_list in str_lists:
        strings.append("".join(str_list))
    return strings


def load_image(img_dir, crop_size = (100,32)):
    """
    :param img_dir:
    :return:img_data
     load image and resize it
    """
    '''
    img = Image.open(img_dir).convert('L')
    size = img.size
    #width = math.ceil(size[0] * (32 / size[1]))
    width = 100
    img = img.resize([width, 32])
    img.show()
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img.size[1], img.size[0], 1))
    '''
    #im_arr = im_arr.astype(np.float32) * (1. / 255) - 0.5
    
    img = cv2.imread(img_dir,0)
    img = cv2.resize(img,crop_size)
    im_arr = np.reshape(img, (img.shape[0], img.shape[1], 1))
    cv2.imshow("im", img)
    
    return im_arr,crop_size[0]

def prepare_data(img_dir):
    """
    :param img_dir:
    :return:
    """
    # first load image and label
    image_raw,width = load_image(img_dir, crop_size = input_size)
    label = load_label_from_img_dir(img_dir)
    label = label.lower()
    return image_raw,label,width


batch_input = tf.placeholder(tf.int32, shape=())
img_input = tf.placeholder(tf.float32, shape=(input_size[1], input_size[0], 1))
img_input = img_input * (1. / 255) - 0.5
img_4d = tf.expand_dims(img_input, 0)


# define the crnn net
#crnn_params = model.CRNNNet.default_params._replace(batch_size=1)._replace(seq_length = SEQ_LEN)._replace(imgH = input_size[1])  # ,seq_length=int(width/4+1)
crnn_params = crnn_params._replace(batch_size=1)
crnn = model.CRNNNet(crnn_params)
logits, inputs, seq_len, W, b = crnn.net(img_4d,batch_input)

decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
val_predict = tf.cast(decoded[0], tf.int32)

saver = tf.train.Saver()

sess = tf.Session()
dir = tf.train.latest_checkpoint(checkpoint_dir)
saver.restore(sess, dir)
sess.run(tf.local_variables_initializer())
print("Model restore!")


def recognize_img(img_dir):
    img_raw,label,width = prepare_data(img_dir)
    
    decoded_s = sess.run([val_predict,log_prob],feed_dict={img_input:img_raw, batch_input:1})
    mean = 0
    for i in range(500):
        t = time.time()
        decoded_s = sess.run([val_predict,log_prob],feed_dict={img_input:img_raw, batch_input:1})
        end_time = time.time() - t
        mean+=end_time
    print("Mean time taken: ", mean/500 )
    # print(decoded_s[0])
    str = sparse_tensor_to_str(decoded_s[0])
    print("Probs: ", decoded_s[1])
    #print("label ",label)
    print('result ',str[0])
    cv2.waitKey(0)



def main(_):
    img_dirs = glob.glob(os.path.join("demo/","*.png"))
    for i,img_dir in enumerate(img_dirs):
        print("index：",i,"name",img_dir)
        #index = int(input("the index choose is :"))
        recognize_img(img_dirs[i])



if __name__ =="__main__":
    tf.app.run()






