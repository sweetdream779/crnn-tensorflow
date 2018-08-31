# crnn-tensorflow
This is fork from [this](https://github.com/AimeeKing/crnn-tensorflow.git) repository. 

This software implements the Convolutional Recurrent Neural Network (CRNN) in tensorflow.Origin software could be found in [crnn](https://github.com/bgshih/crnn)

## run demo

A demo program can be found in `inference.sh`. Before running the demo,you should change paths.
The demo reads an example image and recognizes its text content.

## Train a new model

1. Create .txt file with 
```
path_to_jpg_image path_to_txt_file_iwth_label_for_image
```
contents(see `dataset/train_list.txt` example). Each image contains 1 word and text file with label contains 1 string.

2. Use `dataset/create_dataset.py` to convert your dataset to tfrecord (dont forget change alphabet)

3. Change model parameters in `train.py`

4. Train your model with `train.py`

5. Use `export_model.sh` to freeze your graph.

