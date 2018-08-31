import tensorflow as tf
import model_deploy
# from net import model_save as model
import os
slim = tf.contrib.slim
import time
from net import model
from dataset import read_utils
from tensorflow.python import debug as tf_debug
import numpy as np
from dataset.utils import int_to_char


#######################################################################################################
batch_size = 16
num_readers = 4
num_epochs = 50
model_size = [32,100] #gray
checkpoint_dir = './tmp/'
num_to_show = 10
SEQ_LEN = int(model_size[1]/4 + 1)

valid_num_epochs = 1
valid_batch_size = batch_size

#last symbol for blank output
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz-_"

crnn_params = model.RCNNParams(
            ks=[3, 3, 3, 3, 3, 3, 2],  # kernel_size
            ps = [1, 1, 1, 1, 1, 1, 0], # padding_size
            ss = [1, 1, 1, 1, 1, 1, 1],  # stride_size
            nm = [64, 128, 256, 256, 512, 512, 512],# In/Out size
            leakyRelu = False,
            n_rnn =2,
            nh = 256,#size of the lstm hidden state
            imgH = model_size[0],#the height / width of the input image to network
            nc = 1,##
            nclass = len(alphabet),
            batch_size= batch_size,
            seq_length = SEQ_LEN,
            input_size = 512,
            reuse = tf.AUTO_REUSE)

starter_learning_rate = 0.0001

########################################################################################################

def sparse_tensor_to_str(spares_tensor):
    """
    :param spares_tensor:
    :return: a str
    """
    indices= spares_tensor.indices
    values = spares_tensor.values
    dense_shape = spares_tensor.dense_shape

    number_lists = np.ones(dense_shape,dtype=values.dtype)
    number_lists.fill(-1)
    str_lists = []
    strings = []
    for i,index in enumerate(indices):
        number_lists[index[0],index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([int_to_char(val,alphabet) for val in number_list])
    for str_list in str_lists:
        strings.append("".join(str_list))
    return strings

# =========================================================================== #
# Main
# =========================================================================== #
def run():

    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():

        deploy_config = model_deploy.DeploymentConfig()
        # Create global_step
        global_step = tf.Variable(0,name='global_step',trainable=False)

        file_name = os.path.join("dataset/data", "train2.tfrecords")
        train_images, train_labels, train_path = read_utils.inputs(filename = file_name, batch_size=batch_size,num_epochs=num_epochs, crop_size = model_size)
        
        coord = tf.train.Coordinator()
        crnn = model.CRNNNet(crnn_params)        
        logits, inputs, seq_len, W, b = crnn.net(train_images)

        cost = crnn.losses(train_labels,logits, seq_len)

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 2000, 0.9, staircase=True)
        tf.summary.scalar("learning_rate",learning_rate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss=cost,global_step=global_step)
        optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(loss=cost,global_step=global_step)
        
        tf.summary.scalar("cost",cost)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        pred = tf.cast(decoded[0], tf.int32)
        acc = tf.reduce_mean(tf.edit_distance(pred, train_labels))
        tf.summary.scalar("edit_distance",acc)

        ##################################

        sess = tf.Session()

        save = tf.train.Saver(max_to_keep=20)
        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)            # Start input enqueue threads.
        else:
            save.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
            sess.run(tf.local_variables_initializer())
        
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(checkpoint_dir + 'my-model', sess.graph)

        ##################################
        try:
            step = global_step
            while not coord.should_stop():
                start_time = time.time()

                _, merged_t, train_cost, train_accuracy, lr, step, val_lbls, val_pred, paths = sess.run([optimizer, merged, cost, acc, learning_rate, global_step, train_labels, pred, train_path])
                
                duration = time.time() - start_time

                print("## Step: %d Cost: %.3f"  % (step,train_cost))                
                
                # Print an overview fairly often.
                if step % 10 == 0:                    
                    str_gt = sparse_tensor_to_str(val_lbls)
                    str_pred = sparse_tensor_to_str(val_pred)
                    
                    for i in range(num_to_show):
                        print("  true: ", str_gt[i], "  result: ", str_pred[i], " img_path: ", paths[i])
                    print('Step: %d  train_acc: %.3f (%.3f sec)' % (step, train_accuracy, duration))
                    
                    print('Current lr: %.8f' % lr)
                if step % 100 == 0:
                    save.save(sess, checkpoint_dir +"crnn-model.ckpt", global_step = global_step)
                file_writer.add_summary(merged_t, step)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()
if __name__ == '__main__':
    run()

