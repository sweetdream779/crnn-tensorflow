CHECKPOINT_FILE=/home/irina/openalpr/reps/crnn-tensorflow/tmp_32x100/crnn-model.ckpt-6900
TF_PATH=/home/irina/tensorflow
NAME=crnn
export CUDA_VISIBLE_DEVICES=""
CUR_PATH=${PWD}

##########################################################################################

# Create unfrozen graph with export_inference_graph.py
python3 export_model.py --output_file ${NAME}_unfrozen.pb

python3 ${TF_PATH}/tensorflow/python/tools/freeze_graph.py \
--output_node_names="sparse_tensor_indices,sparse_tensor_values,sparse_tensor_shape,input_plate_size,output_names,input_name,alphabet" \
--input_graph=${NAME}_unfrozen.pb \
--input_checkpoint=${CHECKPOINT_FILE} \
--input_binary=true --output_graph=${NAME}_frozen.pb
