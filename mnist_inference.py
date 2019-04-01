import  tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 通过tf.get_variable函数获取变量
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=1.0))
    # losses集合,add_to_collection张量加入集合
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

# 定义神经网络的前向传播
def inference(input_tensor,regularizer):
    #声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)

    # 声明第二层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    #返回最后前向传播结果
    return  layer2


