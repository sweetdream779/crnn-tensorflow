
N
input_tensorPlaceholder*$
shape:��������� d*
dtype0
2
mul/yConst*
valueB
 *���;*
dtype0
(
mulMulinput_tensormul/y*
T0
2
sub/yConst*
valueB
 *   ?*
dtype0

subSubmulsub/y*
T0
9
input_batch_sizePlaceholder*
dtype0*
shape: 
�
8RCNN_net/conv_0/weights/Initializer/random_uniform/shapeConst*%
valueB"         @   **
_class 
loc:@RCNN_net/conv_0/weights*
dtype0
�
6RCNN_net/conv_0/weights/Initializer/random_uniform/minConst*
valueB
 *�hϽ**
_class 
loc:@RCNN_net/conv_0/weights*
dtype0
�
6RCNN_net/conv_0/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *�h�=**
_class 
loc:@RCNN_net/conv_0/weights
�
@RCNN_net/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_0/weights/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@RCNN_net/conv_0/weights*
dtype0*
seed2 
�
6RCNN_net/conv_0/weights/Initializer/random_uniform/subSub6RCNN_net/conv_0/weights/Initializer/random_uniform/max6RCNN_net/conv_0/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_0/weights
�
6RCNN_net/conv_0/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_0/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_0/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_0/weights
�
2RCNN_net/conv_0/weights/Initializer/random_uniformAdd6RCNN_net/conv_0/weights/Initializer/random_uniform/mul6RCNN_net/conv_0/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_0/weights
�
RCNN_net/conv_0/weights
VariableV2*
shared_name **
_class 
loc:@RCNN_net/conv_0/weights*
dtype0*
	container *
shape:@
�
RCNN_net/conv_0/weights/AssignAssignRCNN_net/conv_0/weights2RCNN_net/conv_0/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@RCNN_net/conv_0/weights*
validate_shape(
v
RCNN_net/conv_0/weights/readIdentityRCNN_net/conv_0/weights*
T0**
_class 
loc:@RCNN_net/conv_0/weights
�
(RCNN_net/conv_0/biases/Initializer/zerosConst*
valueB@*    *)
_class
loc:@RCNN_net/conv_0/biases*
dtype0
�
RCNN_net/conv_0/biases
VariableV2*
	container *
shape:@*
shared_name *)
_class
loc:@RCNN_net/conv_0/biases*
dtype0
�
RCNN_net/conv_0/biases/AssignAssignRCNN_net/conv_0/biases(RCNN_net/conv_0/biases/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@RCNN_net/conv_0/biases*
validate_shape(
s
RCNN_net/conv_0/biases/readIdentityRCNN_net/conv_0/biases*)
_class
loc:@RCNN_net/conv_0/biases*
T0
R
RCNN_net/conv_0/dilation_rateConst*
valueB"      *
dtype0
�
RCNN_net/conv_0/Conv2DConv2DsubRCNN_net/conv_0/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
w
RCNN_net/conv_0/BiasAddBiasAddRCNN_net/conv_0/Conv2DRCNN_net/conv_0/biases/read*
T0*
data_formatNHWC
7
RCNN_net/ReluReluRCNN_net/conv_0/BiasAdd*
T0
�
RCNN_net/pool1/MaxPoolMaxPoolRCNN_net/Relu*
ksize
*
paddingVALID*
T0*
strides
*
data_formatNHWC
�
8RCNN_net/conv_1/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   �   **
_class 
loc:@RCNN_net/conv_1/weights*
dtype0
�
6RCNN_net/conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *�[q�**
_class 
loc:@RCNN_net/conv_1/weights*
dtype0
�
6RCNN_net/conv_1/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *�[q=**
_class 
loc:@RCNN_net/conv_1/weights
�
@RCNN_net/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_1/weights/Initializer/random_uniform/shape*
T0**
_class 
loc:@RCNN_net/conv_1/weights*
dtype0*
seed2 *

seed 
�
6RCNN_net/conv_1/weights/Initializer/random_uniform/subSub6RCNN_net/conv_1/weights/Initializer/random_uniform/max6RCNN_net/conv_1/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_1/weights
�
6RCNN_net/conv_1/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_1/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_1/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_1/weights
�
2RCNN_net/conv_1/weights/Initializer/random_uniformAdd6RCNN_net/conv_1/weights/Initializer/random_uniform/mul6RCNN_net/conv_1/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_1/weights
�
RCNN_net/conv_1/weights
VariableV2**
_class 
loc:@RCNN_net/conv_1/weights*
dtype0*
	container *
shape:@�*
shared_name 
�
RCNN_net/conv_1/weights/AssignAssignRCNN_net/conv_1/weights2RCNN_net/conv_1/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@RCNN_net/conv_1/weights*
validate_shape(
v
RCNN_net/conv_1/weights/readIdentityRCNN_net/conv_1/weights*
T0**
_class 
loc:@RCNN_net/conv_1/weights
�
(RCNN_net/conv_1/biases/Initializer/zerosConst*
valueB�*    *)
_class
loc:@RCNN_net/conv_1/biases*
dtype0
�
RCNN_net/conv_1/biases
VariableV2*
	container *
shape:�*
shared_name *)
_class
loc:@RCNN_net/conv_1/biases*
dtype0
�
RCNN_net/conv_1/biases/AssignAssignRCNN_net/conv_1/biases(RCNN_net/conv_1/biases/Initializer/zeros*
validate_shape(*
use_locking(*
T0*)
_class
loc:@RCNN_net/conv_1/biases
s
RCNN_net/conv_1/biases/readIdentityRCNN_net/conv_1/biases*
T0*)
_class
loc:@RCNN_net/conv_1/biases
R
RCNN_net/conv_1/dilation_rateConst*
valueB"      *
dtype0
�
RCNN_net/conv_1/Conv2DConv2DRCNN_net/pool1/MaxPoolRCNN_net/conv_1/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
w
RCNN_net/conv_1/BiasAddBiasAddRCNN_net/conv_1/Conv2DRCNN_net/conv_1/biases/read*
T0*
data_formatNHWC
9
RCNN_net/Relu_1ReluRCNN_net/conv_1/BiasAdd*
T0
�
RCNN_net/pool2/MaxPoolMaxPoolRCNN_net/Relu_1*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*
T0
�
8RCNN_net/conv_2/weights/Initializer/random_uniform/shapeConst*%
valueB"      �      **
_class 
loc:@RCNN_net/conv_2/weights*
dtype0
�
6RCNN_net/conv_2/weights/Initializer/random_uniform/minConst*
valueB
 *��*�**
_class 
loc:@RCNN_net/conv_2/weights*
dtype0
�
6RCNN_net/conv_2/weights/Initializer/random_uniform/maxConst*
valueB
 *��*=**
_class 
loc:@RCNN_net/conv_2/weights*
dtype0
�
@RCNN_net/conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_2/weights/Initializer/random_uniform/shape*
seed2 *

seed *
T0**
_class 
loc:@RCNN_net/conv_2/weights*
dtype0
�
6RCNN_net/conv_2/weights/Initializer/random_uniform/subSub6RCNN_net/conv_2/weights/Initializer/random_uniform/max6RCNN_net/conv_2/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_2/weights
�
6RCNN_net/conv_2/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_2/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_2/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_2/weights
�
2RCNN_net/conv_2/weights/Initializer/random_uniformAdd6RCNN_net/conv_2/weights/Initializer/random_uniform/mul6RCNN_net/conv_2/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_2/weights
�
RCNN_net/conv_2/weights
VariableV2*
shape:��*
shared_name **
_class 
loc:@RCNN_net/conv_2/weights*
dtype0*
	container 
�
RCNN_net/conv_2/weights/AssignAssignRCNN_net/conv_2/weights2RCNN_net/conv_2/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@RCNN_net/conv_2/weights*
validate_shape(
v
RCNN_net/conv_2/weights/readIdentityRCNN_net/conv_2/weights*
T0**
_class 
loc:@RCNN_net/conv_2/weights
�
(RCNN_net/conv_2/biases/Initializer/zerosConst*
valueB�*    *)
_class
loc:@RCNN_net/conv_2/biases*
dtype0
�
RCNN_net/conv_2/biases
VariableV2*
shared_name *)
_class
loc:@RCNN_net/conv_2/biases*
dtype0*
	container *
shape:�
�
RCNN_net/conv_2/biases/AssignAssignRCNN_net/conv_2/biases(RCNN_net/conv_2/biases/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@RCNN_net/conv_2/biases*
validate_shape(
s
RCNN_net/conv_2/biases/readIdentityRCNN_net/conv_2/biases*
T0*)
_class
loc:@RCNN_net/conv_2/biases
R
RCNN_net/conv_2/dilation_rateConst*
valueB"      *
dtype0
�
RCNN_net/conv_2/Conv2DConv2DRCNN_net/pool2/MaxPoolRCNN_net/conv_2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
w
RCNN_net/conv_2/BiasAddBiasAddRCNN_net/conv_2/Conv2DRCNN_net/conv_2/biases/read*
T0*
data_formatNHWC
K
RCNN_net/batchnorm2/ConstConst*
valueB�*  �?*
dtype0
�
*RCNN_net/batchnorm2/beta/Initializer/zerosConst*
dtype0*
valueB�*    *+
_class!
loc:@RCNN_net/batchnorm2/beta
�
RCNN_net/batchnorm2/beta
VariableV2*
shape:�*
shared_name *+
_class!
loc:@RCNN_net/batchnorm2/beta*
dtype0*
	container 
�
RCNN_net/batchnorm2/beta/AssignAssignRCNN_net/batchnorm2/beta*RCNN_net/batchnorm2/beta/Initializer/zeros*
T0*+
_class!
loc:@RCNN_net/batchnorm2/beta*
validate_shape(*
use_locking(
y
RCNN_net/batchnorm2/beta/readIdentityRCNN_net/batchnorm2/beta*
T0*+
_class!
loc:@RCNN_net/batchnorm2/beta
�
1RCNN_net/batchnorm2/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean*
dtype0
�
RCNN_net/batchnorm2/moving_mean
VariableV2*
dtype0*
	container *
shape:�*
shared_name *2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean
�
&RCNN_net/batchnorm2/moving_mean/AssignAssignRCNN_net/batchnorm2/moving_mean1RCNN_net/batchnorm2/moving_mean/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean*
validate_shape(
�
$RCNN_net/batchnorm2/moving_mean/readIdentityRCNN_net/batchnorm2/moving_mean*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean*
T0
�
4RCNN_net/batchnorm2/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance*
dtype0
�
#RCNN_net/batchnorm2/moving_variance
VariableV2*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance*
dtype0*
	container *
shape:�*
shared_name 
�
*RCNN_net/batchnorm2/moving_variance/AssignAssign#RCNN_net/batchnorm2/moving_variance4RCNN_net/batchnorm2/moving_variance/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance*
validate_shape(
�
(RCNN_net/batchnorm2/moving_variance/readIdentity#RCNN_net/batchnorm2/moving_variance*
T0*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance
D
RCNN_net/batchnorm2/Const_1Const*
valueB *
dtype0
D
RCNN_net/batchnorm2/Const_2Const*
valueB *
dtype0
�
"RCNN_net/batchnorm2/FusedBatchNormFusedBatchNormRCNN_net/conv_2/BiasAddRCNN_net/batchnorm2/ConstRCNN_net/batchnorm2/beta/readRCNN_net/batchnorm2/Const_1RCNN_net/batchnorm2/Const_2*
is_training(*
epsilon%o�:*
T0*
data_formatNHWC
H
RCNN_net/batchnorm2/Const_3Const*
valueB
 *w�?*
dtype0
�
)RCNN_net/batchnorm2/AssignMovingAvg/sub/xConst*
dtype0*
valueB
 *  �?*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean
�
'RCNN_net/batchnorm2/AssignMovingAvg/subSub)RCNN_net/batchnorm2/AssignMovingAvg/sub/xRCNN_net/batchnorm2/Const_3*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean*
T0
�
)RCNN_net/batchnorm2/AssignMovingAvg/sub_1Sub$RCNN_net/batchnorm2/moving_mean/read$RCNN_net/batchnorm2/FusedBatchNorm:1*
T0*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean
�
'RCNN_net/batchnorm2/AssignMovingAvg/mulMul)RCNN_net/batchnorm2/AssignMovingAvg/sub_1'RCNN_net/batchnorm2/AssignMovingAvg/sub*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean*
T0
�
#RCNN_net/batchnorm2/AssignMovingAvg	AssignSubRCNN_net/batchnorm2/moving_mean'RCNN_net/batchnorm2/AssignMovingAvg/mul*
T0*2
_class(
&$loc:@RCNN_net/batchnorm2/moving_mean*
use_locking( 
�
+RCNN_net/batchnorm2/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance*
dtype0
�
)RCNN_net/batchnorm2/AssignMovingAvg_1/subSub+RCNN_net/batchnorm2/AssignMovingAvg_1/sub/xRCNN_net/batchnorm2/Const_3*
T0*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance
�
+RCNN_net/batchnorm2/AssignMovingAvg_1/sub_1Sub(RCNN_net/batchnorm2/moving_variance/read$RCNN_net/batchnorm2/FusedBatchNorm:2*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance*
T0
�
)RCNN_net/batchnorm2/AssignMovingAvg_1/mulMul+RCNN_net/batchnorm2/AssignMovingAvg_1/sub_1)RCNN_net/batchnorm2/AssignMovingAvg_1/sub*
T0*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance
�
%RCNN_net/batchnorm2/AssignMovingAvg_1	AssignSub#RCNN_net/batchnorm2/moving_variance)RCNN_net/batchnorm2/AssignMovingAvg_1/mul*
use_locking( *
T0*6
_class,
*(loc:@RCNN_net/batchnorm2/moving_variance
D
RCNN_net/Relu_2Relu"RCNN_net/batchnorm2/FusedBatchNorm*
T0
�
8RCNN_net/conv_3/weights/Initializer/random_uniform/shapeConst*%
valueB"            **
_class 
loc:@RCNN_net/conv_3/weights*
dtype0
�
6RCNN_net/conv_3/weights/Initializer/random_uniform/minConst*
valueB
 *:��**
_class 
loc:@RCNN_net/conv_3/weights*
dtype0
�
6RCNN_net/conv_3/weights/Initializer/random_uniform/maxConst*
valueB
 *:�=**
_class 
loc:@RCNN_net/conv_3/weights*
dtype0
�
@RCNN_net/conv_3/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_3/weights/Initializer/random_uniform/shape*
T0**
_class 
loc:@RCNN_net/conv_3/weights*
dtype0*
seed2 *

seed 
�
6RCNN_net/conv_3/weights/Initializer/random_uniform/subSub6RCNN_net/conv_3/weights/Initializer/random_uniform/max6RCNN_net/conv_3/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_3/weights
�
6RCNN_net/conv_3/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_3/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_3/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_3/weights
�
2RCNN_net/conv_3/weights/Initializer/random_uniformAdd6RCNN_net/conv_3/weights/Initializer/random_uniform/mul6RCNN_net/conv_3/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_3/weights
�
RCNN_net/conv_3/weights
VariableV2*
shape:��*
shared_name **
_class 
loc:@RCNN_net/conv_3/weights*
dtype0*
	container 
�
RCNN_net/conv_3/weights/AssignAssignRCNN_net/conv_3/weights2RCNN_net/conv_3/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@RCNN_net/conv_3/weights*
validate_shape(
v
RCNN_net/conv_3/weights/readIdentityRCNN_net/conv_3/weights*
T0**
_class 
loc:@RCNN_net/conv_3/weights
�
(RCNN_net/conv_3/biases/Initializer/zerosConst*
valueB�*    *)
_class
loc:@RCNN_net/conv_3/biases*
dtype0
�
RCNN_net/conv_3/biases
VariableV2*
shape:�*
shared_name *)
_class
loc:@RCNN_net/conv_3/biases*
dtype0*
	container 
�
RCNN_net/conv_3/biases/AssignAssignRCNN_net/conv_3/biases(RCNN_net/conv_3/biases/Initializer/zeros*
validate_shape(*
use_locking(*
T0*)
_class
loc:@RCNN_net/conv_3/biases
s
RCNN_net/conv_3/biases/readIdentityRCNN_net/conv_3/biases*
T0*)
_class
loc:@RCNN_net/conv_3/biases
R
RCNN_net/conv_3/dilation_rateConst*
valueB"      *
dtype0
�
RCNN_net/conv_3/Conv2DConv2DRCNN_net/Relu_2RCNN_net/conv_3/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
w
RCNN_net/conv_3/BiasAddBiasAddRCNN_net/conv_3/Conv2DRCNN_net/conv_3/biases/read*
T0*
data_formatNHWC
9
RCNN_net/Relu_3ReluRCNN_net/conv_3/BiasAdd*
T0
l
RCNN_net/pad2d/Pad/paddingsConst*9
value0B."                               *
dtype0
a
RCNN_net/pad2d/PadPadRCNN_net/Relu_3RCNN_net/pad2d/Pad/paddings*
T0*
	Tpaddings0
�
RCNN_net/MaxPool2D/MaxPoolMaxPoolRCNN_net/pad2d/Pad*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*
T0
�
8RCNN_net/conv_4/weights/Initializer/random_uniform/shapeConst*%
valueB"            **
_class 
loc:@RCNN_net/conv_4/weights*
dtype0
�
6RCNN_net/conv_4/weights/Initializer/random_uniform/minConst*
valueB
 *�[�**
_class 
loc:@RCNN_net/conv_4/weights*
dtype0
�
6RCNN_net/conv_4/weights/Initializer/random_uniform/maxConst*
valueB
 *�[�<**
_class 
loc:@RCNN_net/conv_4/weights*
dtype0
�
@RCNN_net/conv_4/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_4/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0**
_class 
loc:@RCNN_net/conv_4/weights
�
6RCNN_net/conv_4/weights/Initializer/random_uniform/subSub6RCNN_net/conv_4/weights/Initializer/random_uniform/max6RCNN_net/conv_4/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_4/weights
�
6RCNN_net/conv_4/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_4/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_4/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_4/weights
�
2RCNN_net/conv_4/weights/Initializer/random_uniformAdd6RCNN_net/conv_4/weights/Initializer/random_uniform/mul6RCNN_net/conv_4/weights/Initializer/random_uniform/min**
_class 
loc:@RCNN_net/conv_4/weights*
T0
�
RCNN_net/conv_4/weights
VariableV2**
_class 
loc:@RCNN_net/conv_4/weights*
dtype0*
	container *
shape:��*
shared_name 
�
RCNN_net/conv_4/weights/AssignAssignRCNN_net/conv_4/weights2RCNN_net/conv_4/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@RCNN_net/conv_4/weights*
validate_shape(
v
RCNN_net/conv_4/weights/readIdentityRCNN_net/conv_4/weights*
T0**
_class 
loc:@RCNN_net/conv_4/weights
�
(RCNN_net/conv_4/biases/Initializer/zerosConst*
valueB�*    *)
_class
loc:@RCNN_net/conv_4/biases*
dtype0
�
RCNN_net/conv_4/biases
VariableV2*
shared_name *)
_class
loc:@RCNN_net/conv_4/biases*
dtype0*
	container *
shape:�
�
RCNN_net/conv_4/biases/AssignAssignRCNN_net/conv_4/biases(RCNN_net/conv_4/biases/Initializer/zeros*
T0*)
_class
loc:@RCNN_net/conv_4/biases*
validate_shape(*
use_locking(
s
RCNN_net/conv_4/biases/readIdentityRCNN_net/conv_4/biases*)
_class
loc:@RCNN_net/conv_4/biases*
T0
R
RCNN_net/conv_4/dilation_rateConst*
dtype0*
valueB"      
�
RCNN_net/conv_4/Conv2DConv2DRCNN_net/MaxPool2D/MaxPoolRCNN_net/conv_4/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
w
RCNN_net/conv_4/BiasAddBiasAddRCNN_net/conv_4/Conv2DRCNN_net/conv_4/biases/read*
data_formatNHWC*
T0
K
RCNN_net/batchnorm4/ConstConst*
valueB�*  �?*
dtype0
�
*RCNN_net/batchnorm4/beta/Initializer/zerosConst*
valueB�*    *+
_class!
loc:@RCNN_net/batchnorm4/beta*
dtype0
�
RCNN_net/batchnorm4/beta
VariableV2*
shared_name *+
_class!
loc:@RCNN_net/batchnorm4/beta*
dtype0*
	container *
shape:�
�
RCNN_net/batchnorm4/beta/AssignAssignRCNN_net/batchnorm4/beta*RCNN_net/batchnorm4/beta/Initializer/zeros*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@RCNN_net/batchnorm4/beta
y
RCNN_net/batchnorm4/beta/readIdentityRCNN_net/batchnorm4/beta*
T0*+
_class!
loc:@RCNN_net/batchnorm4/beta
�
1RCNN_net/batchnorm4/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean*
dtype0
�
RCNN_net/batchnorm4/moving_mean
VariableV2*
shared_name *2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean*
dtype0*
	container *
shape:�
�
&RCNN_net/batchnorm4/moving_mean/AssignAssignRCNN_net/batchnorm4/moving_mean1RCNN_net/batchnorm4/moving_mean/Initializer/zeros*
T0*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean*
validate_shape(*
use_locking(
�
$RCNN_net/batchnorm4/moving_mean/readIdentityRCNN_net/batchnorm4/moving_mean*
T0*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean
�
4RCNN_net/batchnorm4/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance*
dtype0
�
#RCNN_net/batchnorm4/moving_variance
VariableV2*
shared_name *6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance*
dtype0*
	container *
shape:�
�
*RCNN_net/batchnorm4/moving_variance/AssignAssign#RCNN_net/batchnorm4/moving_variance4RCNN_net/batchnorm4/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*
T0*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance
�
(RCNN_net/batchnorm4/moving_variance/readIdentity#RCNN_net/batchnorm4/moving_variance*
T0*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance
D
RCNN_net/batchnorm4/Const_1Const*
valueB *
dtype0
D
RCNN_net/batchnorm4/Const_2Const*
valueB *
dtype0
�
"RCNN_net/batchnorm4/FusedBatchNormFusedBatchNormRCNN_net/conv_4/BiasAddRCNN_net/batchnorm4/ConstRCNN_net/batchnorm4/beta/readRCNN_net/batchnorm4/Const_1RCNN_net/batchnorm4/Const_2*
epsilon%o�:*
T0*
data_formatNHWC*
is_training(
H
RCNN_net/batchnorm4/Const_3Const*
valueB
 *w�?*
dtype0
�
)RCNN_net/batchnorm4/AssignMovingAvg/sub/xConst*
valueB
 *  �?*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean*
dtype0
�
'RCNN_net/batchnorm4/AssignMovingAvg/subSub)RCNN_net/batchnorm4/AssignMovingAvg/sub/xRCNN_net/batchnorm4/Const_3*
T0*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean
�
)RCNN_net/batchnorm4/AssignMovingAvg/sub_1Sub$RCNN_net/batchnorm4/moving_mean/read$RCNN_net/batchnorm4/FusedBatchNorm:1*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean*
T0
�
'RCNN_net/batchnorm4/AssignMovingAvg/mulMul)RCNN_net/batchnorm4/AssignMovingAvg/sub_1'RCNN_net/batchnorm4/AssignMovingAvg/sub*
T0*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean
�
#RCNN_net/batchnorm4/AssignMovingAvg	AssignSubRCNN_net/batchnorm4/moving_mean'RCNN_net/batchnorm4/AssignMovingAvg/mul*
use_locking( *
T0*2
_class(
&$loc:@RCNN_net/batchnorm4/moving_mean
�
+RCNN_net/batchnorm4/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance*
dtype0
�
)RCNN_net/batchnorm4/AssignMovingAvg_1/subSub+RCNN_net/batchnorm4/AssignMovingAvg_1/sub/xRCNN_net/batchnorm4/Const_3*
T0*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance
�
+RCNN_net/batchnorm4/AssignMovingAvg_1/sub_1Sub(RCNN_net/batchnorm4/moving_variance/read$RCNN_net/batchnorm4/FusedBatchNorm:2*
T0*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance
�
)RCNN_net/batchnorm4/AssignMovingAvg_1/mulMul+RCNN_net/batchnorm4/AssignMovingAvg_1/sub_1)RCNN_net/batchnorm4/AssignMovingAvg_1/sub*
T0*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance
�
%RCNN_net/batchnorm4/AssignMovingAvg_1	AssignSub#RCNN_net/batchnorm4/moving_variance)RCNN_net/batchnorm4/AssignMovingAvg_1/mul*
use_locking( *
T0*6
_class,
*(loc:@RCNN_net/batchnorm4/moving_variance
D
RCNN_net/Relu_4Relu"RCNN_net/batchnorm4/FusedBatchNorm*
T0
�
8RCNN_net/conv_5/weights/Initializer/random_uniform/shapeConst*%
valueB"            **
_class 
loc:@RCNN_net/conv_5/weights*
dtype0
�
6RCNN_net/conv_5/weights/Initializer/random_uniform/minConst*
valueB
 *�Ѽ**
_class 
loc:@RCNN_net/conv_5/weights*
dtype0
�
6RCNN_net/conv_5/weights/Initializer/random_uniform/maxConst*
valueB
 *��<**
_class 
loc:@RCNN_net/conv_5/weights*
dtype0
�
@RCNN_net/conv_5/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_5/weights/Initializer/random_uniform/shape*
T0**
_class 
loc:@RCNN_net/conv_5/weights*
dtype0*
seed2 *

seed 
�
6RCNN_net/conv_5/weights/Initializer/random_uniform/subSub6RCNN_net/conv_5/weights/Initializer/random_uniform/max6RCNN_net/conv_5/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_5/weights
�
6RCNN_net/conv_5/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_5/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_5/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_5/weights
�
2RCNN_net/conv_5/weights/Initializer/random_uniformAdd6RCNN_net/conv_5/weights/Initializer/random_uniform/mul6RCNN_net/conv_5/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_5/weights
�
RCNN_net/conv_5/weights
VariableV2*
shape:��*
shared_name **
_class 
loc:@RCNN_net/conv_5/weights*
dtype0*
	container 
�
RCNN_net/conv_5/weights/AssignAssignRCNN_net/conv_5/weights2RCNN_net/conv_5/weights/Initializer/random_uniform*
T0**
_class 
loc:@RCNN_net/conv_5/weights*
validate_shape(*
use_locking(
v
RCNN_net/conv_5/weights/readIdentityRCNN_net/conv_5/weights*
T0**
_class 
loc:@RCNN_net/conv_5/weights
�
(RCNN_net/conv_5/biases/Initializer/zerosConst*
valueB�*    *)
_class
loc:@RCNN_net/conv_5/biases*
dtype0
�
RCNN_net/conv_5/biases
VariableV2*
shape:�*
shared_name *)
_class
loc:@RCNN_net/conv_5/biases*
dtype0*
	container 
�
RCNN_net/conv_5/biases/AssignAssignRCNN_net/conv_5/biases(RCNN_net/conv_5/biases/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@RCNN_net/conv_5/biases*
validate_shape(
s
RCNN_net/conv_5/biases/readIdentityRCNN_net/conv_5/biases*
T0*)
_class
loc:@RCNN_net/conv_5/biases
R
RCNN_net/conv_5/dilation_rateConst*
dtype0*
valueB"      
�
RCNN_net/conv_5/Conv2DConv2DRCNN_net/Relu_4RCNN_net/conv_5/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
w
RCNN_net/conv_5/BiasAddBiasAddRCNN_net/conv_5/Conv2DRCNN_net/conv_5/biases/read*
T0*
data_formatNHWC
9
RCNN_net/Relu_5ReluRCNN_net/conv_5/BiasAdd*
T0
n
RCNN_net/pad2d_1/Pad/paddingsConst*
dtype0*9
value0B."                               
e
RCNN_net/pad2d_1/PadPadRCNN_net/Relu_5RCNN_net/pad2d_1/Pad/paddings*
T0*
	Tpaddings0
�
RCNN_net/MaxPool2D_1/MaxPoolMaxPoolRCNN_net/pad2d_1/Pad*
paddingVALID*
T0*
strides
*
data_formatNHWC*
ksize

�
8RCNN_net/conv_6/weights/Initializer/random_uniform/shapeConst*%
valueB"            **
_class 
loc:@RCNN_net/conv_6/weights*
dtype0
�
6RCNN_net/conv_6/weights/Initializer/random_uniform/minConst*
valueB
 *q��**
_class 
loc:@RCNN_net/conv_6/weights*
dtype0
�
6RCNN_net/conv_6/weights/Initializer/random_uniform/maxConst*
valueB
 *q�=**
_class 
loc:@RCNN_net/conv_6/weights*
dtype0
�
@RCNN_net/conv_6/weights/Initializer/random_uniform/RandomUniformRandomUniform8RCNN_net/conv_6/weights/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@RCNN_net/conv_6/weights*
dtype0*
seed2 
�
6RCNN_net/conv_6/weights/Initializer/random_uniform/subSub6RCNN_net/conv_6/weights/Initializer/random_uniform/max6RCNN_net/conv_6/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_6/weights
�
6RCNN_net/conv_6/weights/Initializer/random_uniform/mulMul@RCNN_net/conv_6/weights/Initializer/random_uniform/RandomUniform6RCNN_net/conv_6/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@RCNN_net/conv_6/weights
�
2RCNN_net/conv_6/weights/Initializer/random_uniformAdd6RCNN_net/conv_6/weights/Initializer/random_uniform/mul6RCNN_net/conv_6/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@RCNN_net/conv_6/weights
�
RCNN_net/conv_6/weights
VariableV2**
_class 
loc:@RCNN_net/conv_6/weights*
dtype0*
	container *
shape:��*
shared_name 
�
RCNN_net/conv_6/weights/AssignAssignRCNN_net/conv_6/weights2RCNN_net/conv_6/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@RCNN_net/conv_6/weights*
validate_shape(
v
RCNN_net/conv_6/weights/readIdentityRCNN_net/conv_6/weights*
T0**
_class 
loc:@RCNN_net/conv_6/weights
�
(RCNN_net/conv_6/biases/Initializer/zerosConst*
valueB�*    *)
_class
loc:@RCNN_net/conv_6/biases*
dtype0
�
RCNN_net/conv_6/biases
VariableV2*
shape:�*
shared_name *)
_class
loc:@RCNN_net/conv_6/biases*
dtype0*
	container 
�
RCNN_net/conv_6/biases/AssignAssignRCNN_net/conv_6/biases(RCNN_net/conv_6/biases/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@RCNN_net/conv_6/biases*
validate_shape(
s
RCNN_net/conv_6/biases/readIdentityRCNN_net/conv_6/biases*
T0*)
_class
loc:@RCNN_net/conv_6/biases
R
RCNN_net/conv_6/dilation_rateConst*
valueB"      *
dtype0
�
RCNN_net/conv_6/Conv2DConv2DRCNN_net/MaxPool2D_1/MaxPoolRCNN_net/conv_6/weights/read*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
w
RCNN_net/conv_6/BiasAddBiasAddRCNN_net/conv_6/Conv2DRCNN_net/conv_6/biases/read*
T0*
data_formatNHWC
K
RCNN_net/batchnorm6/ConstConst*
dtype0*
valueB�*  �?
�
*RCNN_net/batchnorm6/beta/Initializer/zerosConst*
valueB�*    *+
_class!
loc:@RCNN_net/batchnorm6/beta*
dtype0
�
RCNN_net/batchnorm6/beta
VariableV2*
	container *
shape:�*
shared_name *+
_class!
loc:@RCNN_net/batchnorm6/beta*
dtype0
�
RCNN_net/batchnorm6/beta/AssignAssignRCNN_net/batchnorm6/beta*RCNN_net/batchnorm6/beta/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@RCNN_net/batchnorm6/beta*
validate_shape(
y
RCNN_net/batchnorm6/beta/readIdentityRCNN_net/batchnorm6/beta*
T0*+
_class!
loc:@RCNN_net/batchnorm6/beta
�
1RCNN_net/batchnorm6/moving_mean/Initializer/zerosConst*
valueB�*    *2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean*
dtype0
�
RCNN_net/batchnorm6/moving_mean
VariableV2*
shared_name *2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean*
dtype0*
	container *
shape:�
�
&RCNN_net/batchnorm6/moving_mean/AssignAssignRCNN_net/batchnorm6/moving_mean1RCNN_net/batchnorm6/moving_mean/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean*
validate_shape(
�
$RCNN_net/batchnorm6/moving_mean/readIdentityRCNN_net/batchnorm6/moving_mean*
T0*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean
�
4RCNN_net/batchnorm6/moving_variance/Initializer/onesConst*
valueB�*  �?*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance*
dtype0
�
#RCNN_net/batchnorm6/moving_variance
VariableV2*
shape:�*
shared_name *6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance*
dtype0*
	container 
�
*RCNN_net/batchnorm6/moving_variance/AssignAssign#RCNN_net/batchnorm6/moving_variance4RCNN_net/batchnorm6/moving_variance/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance*
validate_shape(
�
(RCNN_net/batchnorm6/moving_variance/readIdentity#RCNN_net/batchnorm6/moving_variance*
T0*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance
D
RCNN_net/batchnorm6/Const_1Const*
valueB *
dtype0
D
RCNN_net/batchnorm6/Const_2Const*
valueB *
dtype0
�
"RCNN_net/batchnorm6/FusedBatchNormFusedBatchNormRCNN_net/conv_6/BiasAddRCNN_net/batchnorm6/ConstRCNN_net/batchnorm6/beta/readRCNN_net/batchnorm6/Const_1RCNN_net/batchnorm6/Const_2*
T0*
data_formatNHWC*
is_training(*
epsilon%o�:
H
RCNN_net/batchnorm6/Const_3Const*
valueB
 *w�?*
dtype0
�
)RCNN_net/batchnorm6/AssignMovingAvg/sub/xConst*
valueB
 *  �?*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean*
dtype0
�
'RCNN_net/batchnorm6/AssignMovingAvg/subSub)RCNN_net/batchnorm6/AssignMovingAvg/sub/xRCNN_net/batchnorm6/Const_3*
T0*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean
�
)RCNN_net/batchnorm6/AssignMovingAvg/sub_1Sub$RCNN_net/batchnorm6/moving_mean/read$RCNN_net/batchnorm6/FusedBatchNorm:1*
T0*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean
�
'RCNN_net/batchnorm6/AssignMovingAvg/mulMul)RCNN_net/batchnorm6/AssignMovingAvg/sub_1'RCNN_net/batchnorm6/AssignMovingAvg/sub*
T0*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean
�
#RCNN_net/batchnorm6/AssignMovingAvg	AssignSubRCNN_net/batchnorm6/moving_mean'RCNN_net/batchnorm6/AssignMovingAvg/mul*
T0*2
_class(
&$loc:@RCNN_net/batchnorm6/moving_mean*
use_locking( 
�
+RCNN_net/batchnorm6/AssignMovingAvg_1/sub/xConst*
valueB
 *  �?*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance*
dtype0
�
)RCNN_net/batchnorm6/AssignMovingAvg_1/subSub+RCNN_net/batchnorm6/AssignMovingAvg_1/sub/xRCNN_net/batchnorm6/Const_3*
T0*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance
�
+RCNN_net/batchnorm6/AssignMovingAvg_1/sub_1Sub(RCNN_net/batchnorm6/moving_variance/read$RCNN_net/batchnorm6/FusedBatchNorm:2*
T0*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance
�
)RCNN_net/batchnorm6/AssignMovingAvg_1/mulMul+RCNN_net/batchnorm6/AssignMovingAvg_1/sub_1)RCNN_net/batchnorm6/AssignMovingAvg_1/sub*
T0*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance
�
%RCNN_net/batchnorm6/AssignMovingAvg_1	AssignSub#RCNN_net/batchnorm6/moving_variance)RCNN_net/batchnorm6/AssignMovingAvg_1/mul*
use_locking( *
T0*6
_class,
*(loc:@RCNN_net/batchnorm6/moving_variance
D
RCNN_net/Relu_6Relu"RCNN_net/batchnorm6/FusedBatchNorm*
T0
L
RCNN_net/SqueezeSqueezeRCNN_net/Relu_6*
squeeze_dims
*
T0
8
RCNN_net/ConstConst*
dtype0*
value	B :
7
RCNN_net/onesConst*
value	B :*
dtype0
;
RCNN_net/mulMulRCNN_net/onesRCNN_net/Const*
T0
A
RCNN_net/ones_1/Less/yConst*
value
B :�*
dtype0
O
RCNN_net/ones_1/LessLessinput_batch_sizeRCNN_net/ones_1/Less/y*
T0
N
RCNN_net/ones_1/packedPackinput_batch_size*
T0*

axis *
N
?
RCNN_net/ones_1/ConstConst*
value	B :*
dtype0
a
RCNN_net/ones_1FillRCNN_net/ones_1/packedRCNN_net/ones_1/Const*

index_type0*
T0
?
RCNN_net/mul_1MulRCNN_net/ones_1RCNN_net/Const*
T0
;
RCNN_net/rnn/RankConst*
value	B :*
dtype0
B
RCNN_net/rnn/range/startConst*
value	B :*
dtype0
B
RCNN_net/rnn/range/deltaConst*
value	B :*
dtype0
n
RCNN_net/rnn/rangeRangeRCNN_net/rnn/range/startRCNN_net/rnn/RankRCNN_net/rnn/range/delta*

Tidx0
Q
RCNN_net/rnn/concat/values_0Const*
valueB"       *
dtype0
B
RCNN_net/rnn/concat/axisConst*
value	B : *
dtype0
�
RCNN_net/rnn/concatConcatV2RCNN_net/rnn/concat/values_0RCNN_net/rnn/rangeRCNN_net/rnn/concat/axis*

Tidx0*
T0*
N
`
RCNN_net/rnn/transpose	TransposeRCNN_net/SqueezeRCNN_net/rnn/concat*
Tperm0*
T0
A
RCNN_net/rnn/sequence_lengthIdentityRCNN_net/mul_1*
T0
L
RCNN_net/rnn/ShapeShapeRCNN_net/rnn/transpose*
T0*
out_type0
N
 RCNN_net/rnn/strided_slice/stackConst*
valueB:*
dtype0
P
"RCNN_net/rnn/strided_slice/stack_1Const*
valueB:*
dtype0
P
"RCNN_net/rnn/strided_slice/stack_2Const*
valueB:*
dtype0
�
RCNN_net/rnn/strided_sliceStridedSliceRCNN_net/rnn/Shape RCNN_net/rnn/strided_slice/stack"RCNN_net/rnn/strided_slice/stack_1"RCNN_net/rnn/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
W
-RCNN_net/rnn/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0
�
)RCNN_net/rnn/LSTMCellZeroState/ExpandDims
ExpandDimsRCNN_net/rnn/strided_slice-RCNN_net/rnn/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0
S
$RCNN_net/rnn/LSTMCellZeroState/ConstConst*
valueB:�*
dtype0
T
*RCNN_net/rnn/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
%RCNN_net/rnn/LSTMCellZeroState/concatConcatV2)RCNN_net/rnn/LSTMCellZeroState/ExpandDims$RCNN_net/rnn/LSTMCellZeroState/Const*RCNN_net/rnn/LSTMCellZeroState/concat/axis*
T0*
N*

Tidx0
W
*RCNN_net/rnn/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
$RCNN_net/rnn/LSTMCellZeroState/zerosFill%RCNN_net/rnn/LSTMCellZeroState/concat*RCNN_net/rnn/LSTMCellZeroState/zeros/Const*
T0*

index_type0
Y
/RCNN_net/rnn/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0
�
+RCNN_net/rnn/LSTMCellZeroState/ExpandDims_1
ExpandDimsRCNN_net/rnn/strided_slice/RCNN_net/rnn/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0
U
&RCNN_net/rnn/LSTMCellZeroState/Const_1Const*
dtype0*
valueB:�
Y
/RCNN_net/rnn/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0
�
+RCNN_net/rnn/LSTMCellZeroState/ExpandDims_2
ExpandDimsRCNN_net/rnn/strided_slice/RCNN_net/rnn/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
U
&RCNN_net/rnn/LSTMCellZeroState/Const_2Const*
dtype0*
valueB:�
V
,RCNN_net/rnn/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0
�
'RCNN_net/rnn/LSTMCellZeroState/concat_1ConcatV2+RCNN_net/rnn/LSTMCellZeroState/ExpandDims_2&RCNN_net/rnn/LSTMCellZeroState/Const_2,RCNN_net/rnn/LSTMCellZeroState/concat_1/axis*
N*

Tidx0*
T0
Y
,RCNN_net/rnn/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
�
&RCNN_net/rnn/LSTMCellZeroState/zeros_1Fill'RCNN_net/rnn/LSTMCellZeroState/concat_1,RCNN_net/rnn/LSTMCellZeroState/zeros_1/Const*

index_type0*
T0
Y
/RCNN_net/rnn/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0
�
+RCNN_net/rnn/LSTMCellZeroState/ExpandDims_3
ExpandDimsRCNN_net/rnn/strided_slice/RCNN_net/rnn/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0
U
&RCNN_net/rnn/LSTMCellZeroState/Const_3Const*
valueB:�*
dtype0
T
RCNN_net/rnn/Shape_1ShapeRCNN_net/rnn/sequence_length*
T0*
out_type0
T
RCNN_net/rnn/stackPackRCNN_net/rnn/strided_slice*
T0*

axis *
N
N
RCNN_net/rnn/EqualEqualRCNN_net/rnn/Shape_1RCNN_net/rnn/stack*
T0
@
RCNN_net/rnn/ConstConst*
valueB: *
dtype0
\
RCNN_net/rnn/AllAllRCNN_net/rnn/EqualRCNN_net/rnn/Const*

Tidx0*
	keep_dims( 
~
RCNN_net/rnn/Assert/ConstConst*M
valueDBB B<Expected shape for Tensor RCNN_net/rnn/sequence_length:0 is *
dtype0
T
RCNN_net/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0
�
!RCNN_net/rnn/Assert/Assert/data_0Const*M
valueDBB B<Expected shape for Tensor RCNN_net/rnn/sequence_length:0 is *
dtype0
Z
!RCNN_net/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0
�
RCNN_net/rnn/Assert/AssertAssertRCNN_net/rnn/All!RCNN_net/rnn/Assert/Assert/data_0RCNN_net/rnn/stack!RCNN_net/rnn/Assert/Assert/data_2RCNN_net/rnn/Shape_1*
T
2*
	summarize
h
RCNN_net/rnn/CheckSeqLenIdentityRCNN_net/rnn/sequence_length^RCNN_net/rnn/Assert/Assert*
T0
N
RCNN_net/rnn/Shape_2ShapeRCNN_net/rnn/transpose*
T0*
out_type0
P
"RCNN_net/rnn/strided_slice_1/stackConst*
valueB: *
dtype0
R
$RCNN_net/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0
R
$RCNN_net/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
RCNN_net/rnn/strided_slice_1StridedSliceRCNN_net/rnn/Shape_2"RCNN_net/rnn/strided_slice_1/stack$RCNN_net/rnn/strided_slice_1/stack_1$RCNN_net/rnn/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
N
RCNN_net/rnn/Shape_3ShapeRCNN_net/rnn/transpose*
out_type0*
T0
P
"RCNN_net/rnn/strided_slice_2/stackConst*
valueB:*
dtype0
R
$RCNN_net/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0
R
$RCNN_net/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
RCNN_net/rnn/strided_slice_2StridedSliceRCNN_net/rnn/Shape_3"RCNN_net/rnn/strided_slice_2/stack$RCNN_net/rnn/strided_slice_2/stack_1$RCNN_net/rnn/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
E
RCNN_net/rnn/ExpandDims/dimConst*
value	B : *
dtype0
u
RCNN_net/rnn/ExpandDims
ExpandDimsRCNN_net/rnn/strided_slice_2RCNN_net/rnn/ExpandDims/dim*

Tdim0*
T0
C
RCNN_net/rnn/Const_1Const*
valueB:�*
dtype0
D
RCNN_net/rnn/concat_1/axisConst*
value	B : *
dtype0
�
RCNN_net/rnn/concat_1ConcatV2RCNN_net/rnn/ExpandDimsRCNN_net/rnn/Const_1RCNN_net/rnn/concat_1/axis*

Tidx0*
T0*
N
E
RCNN_net/rnn/zeros/ConstConst*
valueB
 *    *
dtype0
f
RCNN_net/rnn/zerosFillRCNN_net/rnn/concat_1RCNN_net/rnn/zeros/Const*
T0*

index_type0
B
RCNN_net/rnn/Const_2Const*
valueB: *
dtype0
m
RCNN_net/rnn/MinMinRCNN_net/rnn/CheckSeqLenRCNN_net/rnn/Const_2*
T0*

Tidx0*
	keep_dims( 
B
RCNN_net/rnn/Const_3Const*
valueB: *
dtype0
m
RCNN_net/rnn/MaxMaxRCNN_net/rnn/CheckSeqLenRCNN_net/rnn/Const_3*
T0*

Tidx0*
	keep_dims( 
;
RCNN_net/rnn/timeConst*
dtype0*
value	B : 
�
RCNN_net/rnn/TensorArrayTensorArrayV3RCNN_net/rnn/strided_slice_1*8
tensor_array_name#!RCNN_net/rnn/dynamic_rnn/output_0*
dtype0*%
element_shape:����������*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
RCNN_net/rnn/TensorArray_1TensorArrayV3RCNN_net/rnn/strided_slice_1*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*7
tensor_array_name" RCNN_net/rnn/dynamic_rnn/input_0*
dtype0*%
element_shape:����������
_
%RCNN_net/rnn/TensorArrayUnstack/ShapeShapeRCNN_net/rnn/transpose*
T0*
out_type0
a
3RCNN_net/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
c
5RCNN_net/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
c
5RCNN_net/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
-RCNN_net/rnn/TensorArrayUnstack/strided_sliceStridedSlice%RCNN_net/rnn/TensorArrayUnstack/Shape3RCNN_net/rnn/TensorArrayUnstack/strided_slice/stack5RCNN_net/rnn/TensorArrayUnstack/strided_slice/stack_15RCNN_net/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
U
+RCNN_net/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
U
+RCNN_net/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
%RCNN_net/rnn/TensorArrayUnstack/rangeRange+RCNN_net/rnn/TensorArrayUnstack/range/start-RCNN_net/rnn/TensorArrayUnstack/strided_slice+RCNN_net/rnn/TensorArrayUnstack/range/delta*

Tidx0
�
GRCNN_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3RCNN_net/rnn/TensorArray_1%RCNN_net/rnn/TensorArrayUnstack/rangeRCNN_net/rnn/transposeRCNN_net/rnn/TensorArray_1:1*
T0*)
_class
loc:@RCNN_net/rnn/transpose
@
RCNN_net/rnn/Maximum/xConst*
value	B :*
dtype0
R
RCNN_net/rnn/MaximumMaximumRCNN_net/rnn/Maximum/xRCNN_net/rnn/Max*
T0
\
RCNN_net/rnn/MinimumMinimumRCNN_net/rnn/strided_slice_1RCNN_net/rnn/Maximum*
T0
N
$RCNN_net/rnn/while/iteration_counterConst*
value	B : *
dtype0
�
RCNN_net/rnn/while/EnterEnter$RCNN_net/rnn/while/iteration_counter*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context*
T0*
is_constant( 
�
RCNN_net/rnn/while/Enter_1EnterRCNN_net/rnn/time*
T0*
is_constant( *
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
�
RCNN_net/rnn/while/Enter_2EnterRCNN_net/rnn/TensorArray:1*
is_constant( *
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context*
T0
�
RCNN_net/rnn/while/Enter_3Enter$RCNN_net/rnn/LSTMCellZeroState/zeros*0

frame_name" RCNN_net/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
RCNN_net/rnn/while/Enter_4Enter&RCNN_net/rnn/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
o
RCNN_net/rnn/while/MergeMergeRCNN_net/rnn/while/Enter RCNN_net/rnn/while/NextIteration*
N*
T0
u
RCNN_net/rnn/while/Merge_1MergeRCNN_net/rnn/while/Enter_1"RCNN_net/rnn/while/NextIteration_1*
T0*
N
u
RCNN_net/rnn/while/Merge_2MergeRCNN_net/rnn/while/Enter_2"RCNN_net/rnn/while/NextIteration_2*
T0*
N
u
RCNN_net/rnn/while/Merge_3MergeRCNN_net/rnn/while/Enter_3"RCNN_net/rnn/while/NextIteration_3*
T0*
N
u
RCNN_net/rnn/while/Merge_4MergeRCNN_net/rnn/while/Enter_4"RCNN_net/rnn/while/NextIteration_4*
T0*
N
a
RCNN_net/rnn/while/LessLessRCNN_net/rnn/while/MergeRCNN_net/rnn/while/Less/Enter*
T0
�
RCNN_net/rnn/while/Less/EnterEnterRCNN_net/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
g
RCNN_net/rnn/while/Less_1LessRCNN_net/rnn/while/Merge_1RCNN_net/rnn/while/Less_1/Enter*
T0
�
RCNN_net/rnn/while/Less_1/EnterEnterRCNN_net/rnn/Minimum*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context*
T0*
is_constant(
_
RCNN_net/rnn/while/LogicalAnd
LogicalAndRCNN_net/rnn/while/LessRCNN_net/rnn/while/Less_1
F
RCNN_net/rnn/while/LoopCondLoopCondRCNN_net/rnn/while/LogicalAnd
�
RCNN_net/rnn/while/SwitchSwitchRCNN_net/rnn/while/MergeRCNN_net/rnn/while/LoopCond*
T0*+
_class!
loc:@RCNN_net/rnn/while/Merge
�
RCNN_net/rnn/while/Switch_1SwitchRCNN_net/rnn/while/Merge_1RCNN_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@RCNN_net/rnn/while/Merge_1
�
RCNN_net/rnn/while/Switch_2SwitchRCNN_net/rnn/while/Merge_2RCNN_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@RCNN_net/rnn/while/Merge_2
�
RCNN_net/rnn/while/Switch_3SwitchRCNN_net/rnn/while/Merge_3RCNN_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@RCNN_net/rnn/while/Merge_3
�
RCNN_net/rnn/while/Switch_4SwitchRCNN_net/rnn/while/Merge_4RCNN_net/rnn/while/LoopCond*
T0*-
_class#
!loc:@RCNN_net/rnn/while/Merge_4
M
RCNN_net/rnn/while/IdentityIdentityRCNN_net/rnn/while/Switch:1*
T0
Q
RCNN_net/rnn/while/Identity_1IdentityRCNN_net/rnn/while/Switch_1:1*
T0
Q
RCNN_net/rnn/while/Identity_2IdentityRCNN_net/rnn/while/Switch_2:1*
T0
Q
RCNN_net/rnn/while/Identity_3IdentityRCNN_net/rnn/while/Switch_3:1*
T0
Q
RCNN_net/rnn/while/Identity_4IdentityRCNN_net/rnn/while/Switch_4:1*
T0
`
RCNN_net/rnn/while/add/yConst^RCNN_net/rnn/while/Identity*
value	B :*
dtype0
]
RCNN_net/rnn/while/addAddRCNN_net/rnn/while/IdentityRCNN_net/rnn/while/add/y*
T0
�
$RCNN_net/rnn/while/TensorArrayReadV3TensorArrayReadV3*RCNN_net/rnn/while/TensorArrayReadV3/EnterRCNN_net/rnn/while/Identity_1,RCNN_net/rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
*RCNN_net/rnn/while/TensorArrayReadV3/EnterEnterRCNN_net/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
�
,RCNN_net/rnn/while/TensorArrayReadV3/Enter_1EnterGRCNN_net/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context*
T0*
is_constant(
~
RCNN_net/rnn/while/GreaterEqualGreaterEqualRCNN_net/rnn/while/Identity_1%RCNN_net/rnn/while/GreaterEqual/Enter*
T0
�
%RCNN_net/rnn/while/GreaterEqual/EnterEnterRCNN_net/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
�
>RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"      *0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel*
dtype0
�
<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *�m�*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel*
dtype0
�
<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *�m=*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel*
dtype0
�
FRCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform>RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *

seed *
T0*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel*
dtype0
�
<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/subSub<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/max<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel
�
<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulFRCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniform<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel
�
8RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniformAdd<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/mul<RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel
�
RCNN_net/rnn/lstm_cell/kernel
VariableV2*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel*
dtype0*
	container *
shape:
��*
shared_name 
�
$RCNN_net/rnn/lstm_cell/kernel/AssignAssignRCNN_net/rnn/lstm_cell/kernel8RCNN_net/rnn/lstm_cell/kernel/Initializer/random_uniform*0
_class&
$"loc:@RCNN_net/rnn/lstm_cell/kernel*
validate_shape(*
use_locking(*
T0
V
"RCNN_net/rnn/lstm_cell/kernel/readIdentityRCNN_net/rnn/lstm_cell/kernel*
T0
�
=RCNN_net/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@RCNN_net/rnn/lstm_cell/bias*
dtype0
�
3RCNN_net/rnn/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@RCNN_net/rnn/lstm_cell/bias*
dtype0
�
-RCNN_net/rnn/lstm_cell/bias/Initializer/zerosFill=RCNN_net/rnn/lstm_cell/bias/Initializer/zeros/shape_as_tensor3RCNN_net/rnn/lstm_cell/bias/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@RCNN_net/rnn/lstm_cell/bias
�
RCNN_net/rnn/lstm_cell/bias
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@RCNN_net/rnn/lstm_cell/bias
�
"RCNN_net/rnn/lstm_cell/bias/AssignAssignRCNN_net/rnn/lstm_cell/bias-RCNN_net/rnn/lstm_cell/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@RCNN_net/rnn/lstm_cell/bias
R
 RCNN_net/rnn/lstm_cell/bias/readIdentityRCNN_net/rnn/lstm_cell/bias*
T0
p
(RCNN_net/rnn/while/lstm_cell/concat/axisConst^RCNN_net/rnn/while/Identity*
value	B :*
dtype0
�
#RCNN_net/rnn/while/lstm_cell/concatConcatV2$RCNN_net/rnn/while/TensorArrayReadV3RCNN_net/rnn/while/Identity_4(RCNN_net/rnn/while/lstm_cell/concat/axis*
T0*
N*

Tidx0
�
#RCNN_net/rnn/while/lstm_cell/MatMulMatMul#RCNN_net/rnn/while/lstm_cell/concat)RCNN_net/rnn/while/lstm_cell/MatMul/Enter*
transpose_a( *
transpose_b( *
T0
�
)RCNN_net/rnn/while/lstm_cell/MatMul/EnterEnter"RCNN_net/rnn/lstm_cell/kernel/read*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context*
T0*
is_constant(
�
$RCNN_net/rnn/while/lstm_cell/BiasAddBiasAdd#RCNN_net/rnn/while/lstm_cell/MatMul*RCNN_net/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC
�
*RCNN_net/rnn/while/lstm_cell/BiasAdd/EnterEnter RCNN_net/rnn/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
j
"RCNN_net/rnn/while/lstm_cell/ConstConst^RCNN_net/rnn/while/Identity*
value	B :*
dtype0
t
,RCNN_net/rnn/while/lstm_cell/split/split_dimConst^RCNN_net/rnn/while/Identity*
value	B :*
dtype0
�
"RCNN_net/rnn/while/lstm_cell/splitSplit,RCNN_net/rnn/while/lstm_cell/split/split_dim$RCNN_net/rnn/while/lstm_cell/BiasAdd*
T0*
	num_split
m
"RCNN_net/rnn/while/lstm_cell/add/yConst^RCNN_net/rnn/while/Identity*
valueB
 *  �?*
dtype0
z
 RCNN_net/rnn/while/lstm_cell/addAdd$RCNN_net/rnn/while/lstm_cell/split:2"RCNN_net/rnn/while/lstm_cell/add/y*
T0
Z
$RCNN_net/rnn/while/lstm_cell/SigmoidSigmoid RCNN_net/rnn/while/lstm_cell/add*
T0
u
 RCNN_net/rnn/while/lstm_cell/mulMul$RCNN_net/rnn/while/lstm_cell/SigmoidRCNN_net/rnn/while/Identity_3*
T0
^
&RCNN_net/rnn/while/lstm_cell/Sigmoid_1Sigmoid"RCNN_net/rnn/while/lstm_cell/split*
T0
X
!RCNN_net/rnn/while/lstm_cell/TanhTanh$RCNN_net/rnn/while/lstm_cell/split:1*
T0
}
"RCNN_net/rnn/while/lstm_cell/mul_1Mul&RCNN_net/rnn/while/lstm_cell/Sigmoid_1!RCNN_net/rnn/while/lstm_cell/Tanh*
T0
x
"RCNN_net/rnn/while/lstm_cell/add_1Add RCNN_net/rnn/while/lstm_cell/mul"RCNN_net/rnn/while/lstm_cell/mul_1*
T0
`
&RCNN_net/rnn/while/lstm_cell/Sigmoid_2Sigmoid$RCNN_net/rnn/while/lstm_cell/split:3*
T0
X
#RCNN_net/rnn/while/lstm_cell/Tanh_1Tanh"RCNN_net/rnn/while/lstm_cell/add_1*
T0

"RCNN_net/rnn/while/lstm_cell/mul_2Mul&RCNN_net/rnn/while/lstm_cell/Sigmoid_2#RCNN_net/rnn/while/lstm_cell/Tanh_1*
T0
�
RCNN_net/rnn/while/SelectSelectRCNN_net/rnn/while/GreaterEqualRCNN_net/rnn/while/Select/Enter"RCNN_net/rnn/while/lstm_cell/mul_2*
T0*5
_class+
)'loc:@RCNN_net/rnn/while/lstm_cell/mul_2
�
RCNN_net/rnn/while/Select/EnterEnterRCNN_net/rnn/zeros*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context*
T0*
is_constant(*5
_class+
)'loc:@RCNN_net/rnn/while/lstm_cell/mul_2
�
RCNN_net/rnn/while/Select_1SelectRCNN_net/rnn/while/GreaterEqualRCNN_net/rnn/while/Identity_3"RCNN_net/rnn/while/lstm_cell/add_1*
T0*5
_class+
)'loc:@RCNN_net/rnn/while/lstm_cell/add_1
�
RCNN_net/rnn/while/Select_2SelectRCNN_net/rnn/while/GreaterEqualRCNN_net/rnn/while/Identity_4"RCNN_net/rnn/while/lstm_cell/mul_2*
T0*5
_class+
)'loc:@RCNN_net/rnn/while/lstm_cell/mul_2
�
6RCNN_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3<RCNN_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterRCNN_net/rnn/while/Identity_1RCNN_net/rnn/while/SelectRCNN_net/rnn/while/Identity_2*
T0*5
_class+
)'loc:@RCNN_net/rnn/while/lstm_cell/mul_2
�
<RCNN_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterRCNN_net/rnn/TensorArray*
T0*
is_constant(*5
_class+
)'loc:@RCNN_net/rnn/while/lstm_cell/mul_2*
parallel_iterations *0

frame_name" RCNN_net/rnn/while/while_context
b
RCNN_net/rnn/while/add_1/yConst^RCNN_net/rnn/while/Identity*
value	B :*
dtype0
c
RCNN_net/rnn/while/add_1AddRCNN_net/rnn/while/Identity_1RCNN_net/rnn/while/add_1/y*
T0
R
 RCNN_net/rnn/while/NextIterationNextIterationRCNN_net/rnn/while/add*
T0
V
"RCNN_net/rnn/while/NextIteration_1NextIterationRCNN_net/rnn/while/add_1*
T0
t
"RCNN_net/rnn/while/NextIteration_2NextIteration6RCNN_net/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
Y
"RCNN_net/rnn/while/NextIteration_3NextIterationRCNN_net/rnn/while/Select_1*
T0
Y
"RCNN_net/rnn/while/NextIteration_4NextIterationRCNN_net/rnn/while/Select_2*
T0
C
RCNN_net/rnn/while/ExitExitRCNN_net/rnn/while/Switch*
T0
G
RCNN_net/rnn/while/Exit_1ExitRCNN_net/rnn/while/Switch_1*
T0
G
RCNN_net/rnn/while/Exit_2ExitRCNN_net/rnn/while/Switch_2*
T0
G
RCNN_net/rnn/while/Exit_3ExitRCNN_net/rnn/while/Switch_3*
T0
G
RCNN_net/rnn/while/Exit_4ExitRCNN_net/rnn/while/Switch_4*
T0
�
/RCNN_net/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3RCNN_net/rnn/TensorArrayRCNN_net/rnn/while/Exit_2*+
_class!
loc:@RCNN_net/rnn/TensorArray
�
)RCNN_net/rnn/TensorArrayStack/range/startConst*
value	B : *+
_class!
loc:@RCNN_net/rnn/TensorArray*
dtype0
�
)RCNN_net/rnn/TensorArrayStack/range/deltaConst*
value	B :*+
_class!
loc:@RCNN_net/rnn/TensorArray*
dtype0
�
#RCNN_net/rnn/TensorArrayStack/rangeRange)RCNN_net/rnn/TensorArrayStack/range/start/RCNN_net/rnn/TensorArrayStack/TensorArraySizeV3)RCNN_net/rnn/TensorArrayStack/range/delta*

Tidx0*+
_class!
loc:@RCNN_net/rnn/TensorArray
�
1RCNN_net/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3RCNN_net/rnn/TensorArray#RCNN_net/rnn/TensorArrayStack/rangeRCNN_net/rnn/while/Exit_2*+
_class!
loc:@RCNN_net/rnn/TensorArray*
dtype0*%
element_shape:����������
C
RCNN_net/rnn/Const_4Const*
valueB:�*
dtype0
=
RCNN_net/rnn/Rank_1Const*
dtype0*
value	B :
D
RCNN_net/rnn/range_1/startConst*
value	B :*
dtype0
D
RCNN_net/rnn/range_1/deltaConst*
value	B :*
dtype0
v
RCNN_net/rnn/range_1RangeRCNN_net/rnn/range_1/startRCNN_net/rnn/Rank_1RCNN_net/rnn/range_1/delta*

Tidx0
S
RCNN_net/rnn/concat_2/values_0Const*
dtype0*
valueB"       
D
RCNN_net/rnn/concat_2/axisConst*
dtype0*
value	B : 
�
RCNN_net/rnn/concat_2ConcatV2RCNN_net/rnn/concat_2/values_0RCNN_net/rnn/range_1RCNN_net/rnn/concat_2/axis*
T0*
N*

Tidx0
�
RCNN_net/rnn/transpose_1	Transpose1RCNN_net/rnn/TensorArrayStack/TensorArrayGatherV3RCNN_net/rnn/concat_2*
T0*
Tperm0
K
RCNN_net/Reshape/shapeConst*
valueB"����   *
dtype0
d
RCNN_net/ReshapeReshapeRCNN_net/rnn/transpose_1RCNN_net/Reshape/shape*
T0*
Tshape0
T
RCNN_net/truncated_normal/shapeConst*
dtype0*
valueB"   &   
K
RCNN_net/truncated_normal/meanConst*
dtype0*
valueB
 *    
M
 RCNN_net/truncated_normal/stddevConst*
valueB
 *���=*
dtype0
�
)RCNN_net/truncated_normal/TruncatedNormalTruncatedNormalRCNN_net/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
z
RCNN_net/truncated_normal/mulMul)RCNN_net/truncated_normal/TruncatedNormal RCNN_net/truncated_normal/stddev*
T0
h
RCNN_net/truncated_normalAddRCNN_net/truncated_normal/mulRCNN_net/truncated_normal/mean*
T0
_

RCNN_net/W
VariableV2*
shape:	�&*
shared_name *
dtype0*
	container 
�
RCNN_net/W/AssignAssign
RCNN_net/WRCNN_net/truncated_normal*
use_locking(*
T0*
_class
loc:@RCNN_net/W*
validate_shape(
O
RCNN_net/W/readIdentity
RCNN_net/W*
T0*
_class
loc:@RCNN_net/W
A
RCNN_net/Const_1Const*
valueB&*    *
dtype0
Z

RCNN_net/b
VariableV2*
dtype0*
	container *
shape:&*
shared_name 
�
RCNN_net/b/AssignAssign
RCNN_net/bRCNN_net/Const_1*
validate_shape(*
use_locking(*
T0*
_class
loc:@RCNN_net/b
O
RCNN_net/b/readIdentity
RCNN_net/b*
_class
loc:@RCNN_net/b*
T0
k
RCNN_net/MatMulMatMulRCNN_net/ReshapeRCNN_net/W/read*
transpose_a( *
transpose_b( *
T0
>
RCNN_net/addAddRCNN_net/MatMulRCNN_net/b/read*
T0
M
RCNN_net/Reshape_1/shape/1Const*
valueB :
���������*
dtype0
D
RCNN_net/Reshape_1/shape/2Const*
value	B :&*
dtype0
�
RCNN_net/Reshape_1/shapePackinput_batch_sizeRCNN_net/Reshape_1/shape/1RCNN_net/Reshape_1/shape/2*
N*
T0*

axis 
\
RCNN_net/Reshape_1ReshapeRCNN_net/addRCNN_net/Reshape_1/shape*
T0*
Tshape0
P
RCNN_net/transpose/permConst*!
valueB"          *
dtype0
b
RCNN_net/transpose	TransposeRCNN_net/Reshape_1RCNN_net/transpose/perm*
T0*
Tperm0
�
CTCBeamSearchDecoderCTCBeamSearchDecoderRCNN_net/transposeRCNN_net/mul_1*
merge_repeated( *
	top_paths*

beam_widthd
K
sparse_tensor_indicesCastCTCBeamSearchDecoder*

SrcT0	*

DstT0
L
sparse_tensor_valuesCastCTCBeamSearchDecoder:1*

SrcT0	*

DstT0
K
sparse_tensor_shapeCastCTCBeamSearchDecoder:2*

SrcT0	*

DstT0
I
input_plate_sizeConst*!
valueB"d          *
dtype0
W
alphabetConst*
dtype0*7
value.B, B&0123456789abcdefghijklmnopqrstuvwxyz-_
F
output_namesConst*"
valueBBsparse_tensor*
dtype0
C

input_nameConst*!
valueBBinput_tensor*
dtype0"