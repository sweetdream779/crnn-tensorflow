import tensorflow as tf
import cv2
import numpy as np

def get_image(impath, sess):
	#impath = features['image']

    img_contents = tf.read_file(tf.convert_to_tensor(impath, dtype=tf.string))        
    img = tf.image.decode_jpeg(img_contents, channels=3)

    #img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    #img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    
    img = tf.cast(img, dtype=tf.float32)
    # Extract mean.
    #image = img - img_mean
    #image = img * (1. / 255) - 0.5    
    h, w = [32,100]
    
   # img = image

    initial_width = tf.shape(img)[1]
    initial_height = tf.shape(img)[0]
    

    new_h = tf.to_int32(h)
    ratio = tf.constant(h, dtype=tf.float32)/ tf.to_float(initial_height)
    new_w = ratio * tf.to_float(initial_width)
    new_w = tf.to_int32(new_w)

    #width, height = sess.run([new_w,new_h])
    #print(width, height)
    
    img = tf.image.resize_images(img, [new_h,new_w])
    img = tf.image.pad_to_bounding_box(img, 0, 0, h, tf.maximum(w,new_w))

    img = tf.random_crop(img, [h,w,3])
    img.set_shape([h, w, 3])
    #fwrite = tf.write_file("my_resized_image.jpeg", img)

    img = tf.image.rgb_to_grayscale(img)
    return img, new_w, new_h

with tf.Session() as sess:
	path = "track0001[01].jpg"
	img, new_width, new_height = get_image(path, sess)
	
	im = sess.run(img)
	print(type(im))
	im = np.asarray(im,dtype = np.uint8)
	cv2.imshow("im", im)
	cv2.waitKey(0)
	#print(im)
