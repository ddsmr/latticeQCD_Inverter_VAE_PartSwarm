╫■
к¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8мг
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АЇ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
АЇ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:Ї*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЇЇ*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
ЇЇ*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:Ї*
dtype0
w
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ї*
shared_namez_mean/kernel
p
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes
:	Ї*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
}
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ї*!
shared_namez_log_var/kernel
v
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes
:	Ї*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ь
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╫
value═B╩ B├
▓
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
R
.trainable_variables
/regularization_losses
0	variables
1	keras_api
 
8
0
1
2
3
"4
#5
(6
)7
8
0
1
2
3
"4
#5
(6
)7
н
2layer_metrics
3layer_regularization_losses
	regularization_losses

	variables
4metrics
5non_trainable_variables

6layers
trainable_variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
7layer_metrics
trainable_variables
regularization_losses
	variables
8metrics
9non_trainable_variables

:layers
;layer_regularization_losses
 
 
 
н
<layer_metrics
trainable_variables
regularization_losses
	variables
=metrics
>non_trainable_variables

?layers
@layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
Alayer_metrics
trainable_variables
regularization_losses
	variables
Bmetrics
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
 
 
 
н
Flayer_metrics
trainable_variables
regularization_losses
 	variables
Gmetrics
Hnon_trainable_variables

Ilayers
Jlayer_regularization_losses
YW
VARIABLE_VALUEz_mean/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_mean/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
н
Klayer_metrics
$trainable_variables
%regularization_losses
&	variables
Lmetrics
Mnon_trainable_variables

Nlayers
Olayer_regularization_losses
\Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEz_log_var/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
н
Player_metrics
*trainable_variables
+regularization_losses
,	variables
Qmetrics
Rnon_trainable_variables

Slayers
Tlayer_regularization_losses
 
 
 
н
Ulayer_metrics
.trainable_variables
/regularization_losses
0	variables
Vmetrics
Wnon_trainable_variables

Xlayers
Ylayer_regularization_losses
 
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:         А*
dtype0*
shape:         А
╝
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2	*
Tout
2*M
_output_shapes;
9:         :         :         **
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_6579
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*&
f!R
__inference__traced_save_6970
Ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_restore_7006ль
ы
й
A__inference_dense_1_layer_call_and_return_conditional_losses_6278

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї:::P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л
л
C__inference_z_log_var_layer_call_and_return_conditional_losses_6360

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї:::P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ё
y
$__inference_dense_layer_call_fn_6773

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62212
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
К&
ю
__inference__traced_save_6970
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1e6ce01f56f34bd8965be9ca021bdce7/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╗
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*═
value├B└B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesШ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slicesБ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*_
_input_shapesN
L: :
АЇ:Ї:
ЇЇ:Ї:	Ї::	Ї:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
АЇ:!

_output_shapes	
:Ї:&"
 
_output_shapes
:
ЇЇ:!

_output_shapes	
:Ї:%!

_output_shapes
:	Ї: 

_output_shapes
::%!

_output_shapes
:	Ї: 

_output_shapes
::	

_output_shapes
: 
щ
з
?__inference_dense_layer_call_and_return_conditional_losses_6221

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АЇ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
еO
т
A__inference_encoder_layer_call_and_return_conditional_losses_6648

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Иб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АЇ*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/dropout/ConstЮ
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape═
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2 
dropout/dropout/GreaterEqual/y▀
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/dropout/GreaterEqualШ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/dropout/CastЫ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/dropout/Mul_1з
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02
dense_1/MatMul/ReadVariableOpЯ
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1/MatMulе
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02 
dense_1/BiasAdd/ReadVariableOpв
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
dense_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout_2/dropout/Constж
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout_2/dropout/Mul|
dropout_2/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape╙
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_2/dropout/GreaterEqual/yч
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2 
dropout_2/dropout/GreaterEqualЮ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout_2/dropout/Castг
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout_2/dropout/Mul_1г
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
z_mean/MatMul/ReadVariableOpЭ
z_mean/MatMulMatMuldropout_2/dropout/Mul_1:z:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_mean/MatMulб
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOpЭ
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_mean/BiasAddм
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02!
z_log_var/MatMul/ReadVariableOpж
z_log_var/MatMulMatMuldropout_2/dropout/Mul_1:z:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_log_var/MatMulк
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOpй
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_log_var/BiasAddg
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/ShapeЖ
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stackК
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1К
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2Ш
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicek
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1К
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stackО
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1О
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2д
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1╢
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/meanГ
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sampling/random_normal/stddevА
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:                  *
dtype0*
seed▒ х)*
seed2┬ЗЇ2-
+sampling/random_normal/RandomStandardNormal╪
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:                  2
sampling/random_normal/mul╕
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:                  2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/xК
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:         2
sampling/ExpЗ
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:         2
sampling/mul_1Д
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:         2
sampling/addk
IdentityIdentityz_mean/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identityr

Identity_1Identityz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_1h

Identity_2Identitysampling/add:z:0*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А:::::::::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
є)
 
A__inference_encoder_layer_call_and_return_conditional_losses_6414
input_1

dense_6232

dense_6234
dense_1_6289
dense_1_6291
z_mean_6345
z_mean_6347
z_log_var_6371
z_log_var_6373
identity

identity_1

identity_2Ивdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв sampling/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall▀
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_6232
dense_6234*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62212
dense/StatefulPartitionedCallц
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_62492!
dropout/StatefulPartitionedCallК
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_6289dense_1_6291*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62782!
dense_1/StatefulPartitionedCallР
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_63062#
!dropout_2/StatefulPartitionedCallЖ
z_mean/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0z_mean_6345z_mean_6347*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_63342 
z_mean/StatefulPartitionedCallХ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0z_log_var_6371z_log_var_6373*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_63602#
!z_log_var/StatefulPartitionedCall║
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_sampling_layer_call_and_return_conditional_losses_64022"
 sampling/StatefulPartitionedCallы
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityЄ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1ё

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Л
л
C__inference_z_log_var_layer_call_and_return_conditional_losses_6876

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї:::P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
з
?__inference_dense_layer_call_and_return_conditional_losses_6764

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АЇ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
И
и
@__inference_z_mean_layer_call_and_return_conditional_losses_6857

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї:::P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ё)
■
A__inference_encoder_layer_call_and_return_conditional_losses_6475

inputs

dense_6449

dense_6451
dense_1_6455
dense_1_6457
z_mean_6461
z_mean_6463
z_log_var_6466
z_log_var_6468
identity

identity_1

identity_2Ивdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв sampling/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall▐
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_6449
dense_6451*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62212
dense/StatefulPartitionedCallц
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_62492!
dropout/StatefulPartitionedCallК
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_6455dense_1_6457*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62782!
dense_1/StatefulPartitionedCallР
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_63062#
!dropout_2/StatefulPartitionedCallЖ
z_mean/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0z_mean_6461z_mean_6463*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_63342 
z_mean/StatefulPartitionedCallХ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0z_log_var_6466z_log_var_6468*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_63602#
!z_log_var/StatefulPartitionedCall║
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_sampling_layer_call_and_return_conditional_losses_64022"
 sampling/StatefulPartitionedCallы
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityЄ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1ё

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_6837

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         Ї2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
И
и
@__inference_z_mean_layer_call_and_return_conditional_losses_6334

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї:::P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ў<
т
A__inference_encoder_layer_call_and_return_conditional_losses_6703

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Иб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АЇ*
dtype02
dense/MatMul/ReadVariableOpЖ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:         Ї2
dropout/Identityз
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02
dense_1/MatMul/ReadVariableOpЯ
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1/MatMulе
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02 
dense_1/BiasAdd/ReadVariableOpв
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
dense_1/ReluГ
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*(
_output_shapes
:         Ї2
dropout_2/Identityг
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02
z_mean/MatMul/ReadVariableOpЭ
z_mean/MatMulMatMuldropout_2/Identity:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_mean/MatMulб
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOpЭ
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_mean/BiasAddм
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02!
z_log_var/MatMul/ReadVariableOpж
z_log_var/MatMulMatMuldropout_2/Identity:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_log_var/MatMulк
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOpй
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
z_log_var/BiasAddg
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/ShapeЖ
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stackК
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1К
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2Ш
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicek
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1К
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stackО
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1О
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2д
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1╢
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/meanГ
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sampling/random_normal/stddevА
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:                  *
dtype0*
seed▒ х)*
seed2└ўў2-
+sampling/random_normal/RandomStandardNormal╪
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:                  2
sampling/random_normal/mul╕
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:                  2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/xК
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:         2
sampling/ExpЗ
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:         2
sampling/mul_1Д
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:         2
sampling/addk
IdentityIdentityz_mean/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identityr

Identity_1Identityz_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_1h

Identity_2Identitysampling/add:z:0*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А:::::::::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Є
D
(__inference_dropout_2_layer_call_fn_6847

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_63112
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
╩
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_6311

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         Ї2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
в
Ў
&__inference_encoder_layer_call_fn_6552
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2ИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*M
_output_shapes;
9:         :         :         **
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_65292
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ж
`
A__inference_dropout_layer_call_and_return_conditional_losses_6249

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
╚
_
A__inference_dropout_layer_call_and_return_conditional_losses_6790

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         Ї2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
ю
B
&__inference_dropout_layer_call_fn_6800

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_62542
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
И
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_6832

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
╛%
╕
A__inference_encoder_layer_call_and_return_conditional_losses_6529

inputs

dense_6503

dense_6505
dense_1_6509
dense_1_6511
z_mean_6515
z_mean_6517
z_log_var_6520
z_log_var_6522
identity

identity_1

identity_2Ивdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв sampling/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall▐
dense/StatefulPartitionedCallStatefulPartitionedCallinputs
dense_6503
dense_6505*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62212
dense/StatefulPartitionedCall╬
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_62542
dropout/PartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_6509dense_1_6511*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62782!
dense_1/StatefulPartitionedCall╓
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_63112
dropout_2/PartitionedCall■
z_mean/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0z_mean_6515z_mean_6517*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_63342 
z_mean/StatefulPartitionedCallН
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0z_log_var_6520z_log_var_6522*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_63602#
!z_log_var/StatefulPartitionedCallЦ
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_sampling_layer_call_and_return_conditional_losses_64022"
 sampling/StatefulPartitionedCallе
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityм

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1л

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
К
q
B__inference_sampling_layer_call_and_return_conditional_losses_6911
inputs_0
inputs_1
identityИF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Т
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
random_normal/stddevх
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:                  *
dtype0*
seed▒ х)*
seed2Длс2$
"random_normal/RandomStandardNormal┤
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:                  2
random_normal/mulФ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:                  2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:         2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:         2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
¤
o
B__inference_sampling_layer_call_and_return_conditional_losses_6402

inputs
inputs_1
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Т
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
random_normal/stddevф
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:                  *
dtype0*
seed▒ х)*
seed2╟╥-2$
"random_normal/RandomStandardNormal┤
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:                  2
random_normal/mulФ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:                  2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:         2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:         2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
в
Ў
&__inference_encoder_layer_call_fn_6498
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2ИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*M
_output_shapes;
9:         :         :         **
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_64752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▄E
Б
__inference__wrapped_model_6206
input_10
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2И╣
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
АЇ*
dtype02%
#encoder/dense/MatMul/ReadVariableOpЯ
encoder/dense/MatMulMatMulinput_1+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
encoder/dense/MatMul╖
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp║
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
encoder/dense/BiasAddГ
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
encoder/dense/ReluХ
encoder/dropout/IdentityIdentity encoder/dense/Relu:activations:0*
T0*(
_output_shapes
:         Ї2
encoder/dropout/Identity┐
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp┐
encoder/dense_1/MatMulMatMul!encoder/dropout/Identity:output:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
encoder/dense_1/MatMul╜
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp┬
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
encoder/dense_1/BiasAddЙ
encoder/dense_1/ReluRelu encoder/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
encoder/dense_1/ReluЫ
encoder/dropout_2/IdentityIdentity"encoder/dense_1/Relu:activations:0*
T0*(
_output_shapes
:         Ї2
encoder/dropout_2/Identity╗
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02&
$encoder/z_mean/MatMul/ReadVariableOp╜
encoder/z_mean/MatMulMatMul#encoder/dropout_2/Identity:output:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
encoder/z_mean/MatMul╣
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/z_mean/BiasAdd/ReadVariableOp╜
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
encoder/z_mean/BiasAdd─
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	Ї*
dtype02)
'encoder/z_log_var/MatMul/ReadVariableOp╞
encoder/z_log_var/MatMulMatMul#encoder/dropout_2/Identity:output:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
encoder/z_log_var/MatMul┬
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/z_log_var/BiasAdd/ReadVariableOp╔
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
encoder/z_log_var/BiasAdd
encoder/sampling/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling/ShapeЦ
$encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$encoder/sampling/strided_slice/stackЪ
&encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder/sampling/strided_slice/stack_1Ъ
&encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&encoder/sampling/strided_slice/stack_2╚
encoder/sampling/strided_sliceStridedSliceencoder/sampling/Shape:output:0-encoder/sampling/strided_slice/stack:output:0/encoder/sampling/strided_slice/stack_1:output:0/encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
encoder/sampling/strided_sliceГ
encoder/sampling/Shape_1Shapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling/Shape_1Ъ
&encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&encoder/sampling/strided_slice_1/stackЮ
(encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling/strided_slice_1/stack_1Ю
(encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(encoder/sampling/strided_slice_1/stack_2╘
 encoder/sampling/strided_slice_1StridedSlice!encoder/sampling/Shape_1:output:0/encoder/sampling/strided_slice_1/stack:output:01encoder/sampling/strided_slice_1/stack_1:output:01encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 encoder/sampling/strided_slice_1╓
$encoder/sampling/random_normal/shapePack'encoder/sampling/strided_slice:output:0)encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2&
$encoder/sampling/random_normal/shapeП
#encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder/sampling/random_normal/meanУ
%encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%encoder/sampling/random_normal/stddevШ
3encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal-encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:                  *
dtype0*
seed▒ х)*
seed2└┌╠25
3encoder/sampling/random_normal/RandomStandardNormal°
"encoder/sampling/random_normal/mulMul<encoder/sampling/random_normal/RandomStandardNormal:output:0.encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:                  2$
"encoder/sampling/random_normal/mul╪
encoder/sampling/random_normalAdd&encoder/sampling/random_normal/mul:z:0,encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:                  2 
encoder/sampling/random_normalu
encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling/mul/xк
encoder/sampling/mulMulencoder/sampling/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2
encoder/sampling/mul
encoder/sampling/ExpExpencoder/sampling/mul:z:0*
T0*'
_output_shapes
:         2
encoder/sampling/Expз
encoder/sampling/mul_1Mulencoder/sampling/Exp:y:0"encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:         2
encoder/sampling/mul_1д
encoder/sampling/addAddV2encoder/z_mean/BiasAdd:output:0encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:         2
encoder/sampling/addl
IdentityIdentityencoder/sampling/add:z:0*
T0*'
_output_shapes
:         2

Identityz

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_1w

Identity_2Identityencoder/z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А:::::::::Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
a
(__inference_dropout_2_layer_call_fn_6842

inputs
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_63062
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ё
z
%__inference_z_mean_layer_call_fn_6866

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_63342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ж
`
A__inference_dropout_layer_call_and_return_conditional_losses_6785

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ы*
▓
 __inference__traced_restore_7006
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias$
 assignvariableop_4_z_mean_kernel"
assignvariableop_5_z_mean_bias'
#assignvariableop_6_z_log_var_kernel%
!assignvariableop_7_z_log_var_bias

identity_9ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7в	RestoreV2вRestoreV2_1┴
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*═
value├B└B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЮ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices╙
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityН
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1У
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ч
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Х
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ц
AssignVariableOp_4AssignVariableOp assignvariableop_4_z_mean_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ф
AssignVariableOp_5AssignVariableOpassignvariableop_5_z_mean_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Щ
AssignVariableOp_6AssignVariableOp#assignvariableop_6_z_log_var_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ч
AssignVariableOp_7AssignVariableOp!assignvariableop_7_z_log_var_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpО

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8Ъ

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╚
_
A__inference_dropout_layer_call_and_return_conditional_losses_6254

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         Ї2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         Ї2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
№
Є
"__inference_signature_wrapper_6579
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2ИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*M
_output_shapes;
9:         :         :         **
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__wrapped_model_62062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ў
}
(__inference_z_log_var_layer_call_fn_6885

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_63602
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¤
p
'__inference_sampling_layer_call_fn_6917
inputs_0
inputs_1
identityИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_sampling_layer_call_and_return_conditional_losses_64022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Я
ї
&__inference_encoder_layer_call_fn_6753

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2ИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*M
_output_shapes;
9:         :         :         **
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_65292
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┴%
╣
A__inference_encoder_layer_call_and_return_conditional_losses_6443
input_1

dense_6417

dense_6419
dense_1_6423
dense_1_6425
z_mean_6429
z_mean_6431
z_log_var_6434
z_log_var_6436
identity

identity_1

identity_2Ивdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв sampling/StatefulPartitionedCallв!z_log_var/StatefulPartitionedCallвz_mean/StatefulPartitionedCall▀
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1
dense_6417
dense_6419*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_62212
dense/StatefulPartitionedCall╬
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_62542
dropout/PartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_6423dense_1_6425*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62782!
dense_1/StatefulPartitionedCall╓
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_63112
dropout_2/PartitionedCall■
z_mean/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0z_mean_6429z_mean_6431*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_63342 
z_mean/StatefulPartitionedCallН
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0z_log_var_6434z_log_var_6436*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_63602#
!z_log_var/StatefulPartitionedCallЦ
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_sampling_layer_call_and_return_conditional_losses_64022"
 sampling/StatefulPartitionedCallе
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityм

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1л

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ї
{
&__inference_dense_1_layer_call_fn_6820

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         Ї*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_62782
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Я
ї
&__inference_encoder_layer_call_fn_6728

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2ИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*M
_output_shapes;
9:         :         :         **
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_64752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1Т

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*G
_input_shapes6
4:         А::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
·
_
&__inference_dropout_layer_call_fn_6795

inputs
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         Ї* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_62492
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
ы
й
A__inference_dense_1_layer_call_and_return_conditional_losses_6811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЇЇ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ї2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Ї2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ї:::P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
И
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_6306

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         Ї2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         Ї*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Ї2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Ї2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         Ї2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         Ї2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ї:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*з
serving_defaultУ
<
input_11
serving_default_input_1:0         А<
sampling0
StatefulPartitionedCall:0         =
	z_log_var0
StatefulPartitionedCall:1         :
z_mean0
StatefulPartitionedCall:2         tensorflow/serving/predict:Э╙
Ў3
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
*Z&call_and_return_all_conditional_losses
[_default_save_signature
\__call__"ъ0
_tf_keras_model╨0{"class_name": "Model", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2304]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2304]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}}}
я"ь
_tf_keras_input_layer╠{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2304]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2304]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
╬

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"й
_tf_keras_layerП{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
╛
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"п
_tf_keras_layerХ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
╨

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"л
_tf_keras_layerС{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
┬
trainable_variables
regularization_losses
 	variables
!	keras_api
*c&call_and_return_all_conditional_losses
d__call__"│
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
╬

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
*e&call_and_return_all_conditional_losses
f__call__"й
_tf_keras_layerП{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
╘

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
*g&call_and_return_all_conditional_losses
h__call__"п
_tf_keras_layerХ{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}
Т
.trainable_variables
/regularization_losses
0	variables
1	keras_api
*i&call_and_return_all_conditional_losses
j__call__"Г
_tf_keras_layerщ{"class_name": "Sampling", "name": "sampling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "sampling", "trainable": true, "dtype": "float32"}}
 "
trackable_list_wrapper
X
0
1
2
3
"4
#5
(6
)7"
trackable_list_wrapper
X
0
1
2
3
"4
#5
(6
)7"
trackable_list_wrapper
╩
2layer_metrics
3layer_regularization_losses
	regularization_losses

	variables
4metrics
5non_trainable_variables

6layers
trainable_variables
\__call__
[_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
kserving_default"
signature_map
 :
АЇ2dense/kernel
:Ї2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
7layer_metrics
trainable_variables
regularization_losses
	variables
8metrics
9non_trainable_variables

:layers
;layer_regularization_losses
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
<layer_metrics
trainable_variables
regularization_losses
	variables
=metrics
>non_trainable_variables

?layers
@layer_regularization_losses
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
": 
ЇЇ2dense_1/kernel
:Ї2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
Alayer_metrics
trainable_variables
regularization_losses
	variables
Bmetrics
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Flayer_metrics
trainable_variables
regularization_losses
 	variables
Gmetrics
Hnon_trainable_variables

Ilayers
Jlayer_regularization_losses
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 :	Ї2z_mean/kernel
:2z_mean/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
н
Klayer_metrics
$trainable_variables
%regularization_losses
&	variables
Lmetrics
Mnon_trainable_variables

Nlayers
Olayer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
#:!	Ї2z_log_var/kernel
:2z_log_var/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
н
Player_metrics
*trainable_variables
+regularization_losses
,	variables
Qmetrics
Rnon_trainable_variables

Slayers
Tlayer_regularization_losses
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Ulayer_metrics
.trainable_variables
/regularization_losses
0	variables
Vmetrics
Wnon_trainable_variables

Xlayers
Ylayer_regularization_losses
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╥2╧
A__inference_encoder_layer_call_and_return_conditional_losses_6648
A__inference_encoder_layer_call_and_return_conditional_losses_6703
A__inference_encoder_layer_call_and_return_conditional_losses_6443
A__inference_encoder_layer_call_and_return_conditional_losses_6414└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
__inference__wrapped_model_6206╖
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *'в$
"К
input_1         А
ц2у
&__inference_encoder_layer_call_fn_6498
&__inference_encoder_layer_call_fn_6552
&__inference_encoder_layer_call_fn_6728
&__inference_encoder_layer_call_fn_6753└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_6764в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_dense_layer_call_fn_6773в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
└2╜
A__inference_dropout_layer_call_and_return_conditional_losses_6790
A__inference_dropout_layer_call_and_return_conditional_losses_6785┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
&__inference_dropout_layer_call_fn_6800
&__inference_dropout_layer_call_fn_6795┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ы2ш
A__inference_dense_1_layer_call_and_return_conditional_losses_6811в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_1_layer_call_fn_6820в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─2┴
C__inference_dropout_2_layer_call_and_return_conditional_losses_6837
C__inference_dropout_2_layer_call_and_return_conditional_losses_6832┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
О2Л
(__inference_dropout_2_layer_call_fn_6842
(__inference_dropout_2_layer_call_fn_6847┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
@__inference_z_mean_layer_call_and_return_conditional_losses_6857в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_z_mean_layer_call_fn_6866в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_z_log_var_layer_call_and_return_conditional_losses_6876в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_z_log_var_layer_call_fn_6885в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_sampling_layer_call_and_return_conditional_losses_6911в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_sampling_layer_call_fn_6917в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
1B/
"__inference_signature_wrapper_6579input_1Ў
__inference__wrapped_model_6206╥"#()1в.
'в$
"К
input_1         А
к "ТкО
.
sampling"К
sampling         
0
	z_log_var#К 
	z_log_var         
*
z_mean К
z_mean         г
A__inference_dense_1_layer_call_and_return_conditional_losses_6811^0в-
&в#
!К
inputs         Ї
к "&в#
К
0         Ї
Ъ {
&__inference_dense_1_layer_call_fn_6820Q0в-
&в#
!К
inputs         Ї
к "К         Їб
?__inference_dense_layer_call_and_return_conditional_losses_6764^0в-
&в#
!К
inputs         А
к "&в#
К
0         Ї
Ъ y
$__inference_dense_layer_call_fn_6773Q0в-
&в#
!К
inputs         А
к "К         Їе
C__inference_dropout_2_layer_call_and_return_conditional_losses_6832^4в1
*в'
!К
inputs         Ї
p
к "&в#
К
0         Ї
Ъ е
C__inference_dropout_2_layer_call_and_return_conditional_losses_6837^4в1
*в'
!К
inputs         Ї
p 
к "&в#
К
0         Ї
Ъ }
(__inference_dropout_2_layer_call_fn_6842Q4в1
*в'
!К
inputs         Ї
p
к "К         Ї}
(__inference_dropout_2_layer_call_fn_6847Q4в1
*в'
!К
inputs         Ї
p 
к "К         Їг
A__inference_dropout_layer_call_and_return_conditional_losses_6785^4в1
*в'
!К
inputs         Ї
p
к "&в#
К
0         Ї
Ъ г
A__inference_dropout_layer_call_and_return_conditional_losses_6790^4в1
*в'
!К
inputs         Ї
p 
к "&в#
К
0         Ї
Ъ {
&__inference_dropout_layer_call_fn_6795Q4в1
*в'
!К
inputs         Ї
p
к "К         Ї{
&__inference_dropout_layer_call_fn_6800Q4в1
*в'
!К
inputs         Ї
p 
к "К         Їў
A__inference_encoder_layer_call_and_return_conditional_losses_6414▒"#()9в6
/в,
"К
input_1         А
p

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ ў
A__inference_encoder_layer_call_and_return_conditional_losses_6443▒"#()9в6
/в,
"К
input_1         А
p 

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ Ў
A__inference_encoder_layer_call_and_return_conditional_losses_6648░"#()8в5
.в+
!К
inputs         А
p

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ Ў
A__inference_encoder_layer_call_and_return_conditional_losses_6703░"#()8в5
.в+
!К
inputs         А
p 

 
к "jвg
`Ъ]
К
0/0         
К
0/1         
К
0/2         
Ъ ╠
&__inference_encoder_layer_call_fn_6498б"#()9в6
/в,
"К
input_1         А
p

 
к "ZЪW
К
0         
К
1         
К
2         ╠
&__inference_encoder_layer_call_fn_6552б"#()9в6
/в,
"К
input_1         А
p 

 
к "ZЪW
К
0         
К
1         
К
2         ╦
&__inference_encoder_layer_call_fn_6728а"#()8в5
.в+
!К
inputs         А
p

 
к "ZЪW
К
0         
К
1         
К
2         ╦
&__inference_encoder_layer_call_fn_6753а"#()8в5
.в+
!К
inputs         А
p 

 
к "ZЪW
К
0         
К
1         
К
2         ╩
B__inference_sampling_layer_call_and_return_conditional_losses_6911ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "%в"
К
0         
Ъ б
'__inference_sampling_layer_call_fn_6917vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "К         Д
"__inference_signature_wrapper_6579▌"#()<в9
в 
2к/
-
input_1"К
input_1         А"ТкО
.
sampling"К
sampling         
0
	z_log_var#К 
	z_log_var         
*
z_mean К
z_mean         д
C__inference_z_log_var_layer_call_and_return_conditional_losses_6876]()0в-
&в#
!К
inputs         Ї
к "%в"
К
0         
Ъ |
(__inference_z_log_var_layer_call_fn_6885P()0в-
&в#
!К
inputs         Ї
к "К         б
@__inference_z_mean_layer_call_and_return_conditional_losses_6857]"#0в-
&в#
!К
inputs         Ї
к "%в"
К
0         
Ъ y
%__inference_z_mean_layer_call_fn_6866P"#0в-
&в#
!К
inputs         Ї
к "К         