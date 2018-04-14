# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:21:06 2018

@author: yanghe
"""

from tensorflow.examples.tutorials.mnist import input_data  
import tensorflow as tf  
  
mnist = input_data.read_data_sets(r'E:\python\mnist_data', one_hot=True)  
  
############################################################################################  
# tf.InteractiveSession 配置完了以后，由tf.Tensor.eval或tf.Operation.run 的方式来生效  
############################################################################################  
sess = tf.InteractiveSession()  
  
''''' 
weight_variable     权重变量 
@:param shape       形状 
@:return Variable   返回变量 
 
通过给定形状，标准差的正态分布生成一个随机值 
由于此返回结果需要后续不断变更，此处返回类型为tf.Variable 
'''  
def weight_variable(shape):  
    ############################################################################################  
    # tf.truncated_normal 截尾正态分布  
    # 参数 shape 形状           是一个Tensor或是python数组，用来表示正态分布有几维和对应维有几个特征，比如：[3, 4]表示有两维，第一维有3个特征，第二维有4个特征  
    # 参数 stddev 标准差         默认是1  
    # 返回 指定形状               指定标准差的正态分布中的一个随机值生成的一个Tensor  
    ############################################################################################  
    initial = tf.truncated_normal(shape,stddev=0.1)  
    ############################################################################################  
    # tf.Variable 变量定义  变量是构造流程图的基本元素  
    # 参数 initial_value 值            任何类型的值都可以  
    # 返回 tf.Variable  
    ############################################################################################  
    return tf.Variable(initial)  
  
''''' 
bias_variable       偏移变量 
@:param shape       形状 
@:return Variable   返回变量 
'''  
def bias_variable(shape):  
    ############################################################################################  
    # tf.constant 常量定义  
    # 参数 value 值            默认返回当前值，如果shape有定义的话，就是一个由此值形成的多维数组  
    # 参数 shape 形状           默认是None  
    # 返回 Tensor  
    ############################################################################################  
    initial = tf.constant(0.1,shape=shape)  
    return tf.Variable(initial)  
  
''''' 
conv2d              卷积 
@:param x           输入图像 
@:param w           卷积核 
@:return Tensor     做过卷积后的图像 
'''  
def conv2d(x, w):  
    ############################################################################################  
    # tf.nn.conv2d 卷积函数  
    # 参数 input 输入图像             输入数据也是四维[图片数量, 图片高度, 图片宽度, 图像通道数][batch, height, width, channels]  
    # 参数 filter 卷积核             四维[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]  
    # 参数 strides 卷积核移动量     四维[图片数量, 图片高度, 图片宽度, 图像通道数]  
    # 参数 padding 边缘处理方式     SAME和VALID,SAME就是可以在外围补0再卷积，VALID不能补0  
    # 返回 Tensor  
    ############################################################################################  
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')  
  
''''' 
max_pool_2x2       用2x2模板做池化 
@:param x          需要池化的对象 
@:return Tensor    返回 
'''  
def max_pool_2x2(x):  
    ############################################################################################  
    # tf.nn.conv2d 卷积函数  
    # 参数 value 输入图像             四维[图片数量, 图片高度, 图片宽度, 图像通道数]  
    # 参数 ksize 池化窗口             四维[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]  
    # 参数 strides 卷积核移动量     四维[图片数量, 图片高度, 图片宽度, 图像通道数]，一般不对图片数量和图像通道数进行池化，所以都是1  
    # 参数 padding 边缘处理方式     SAME和VALID,SAME就是可以在外围补0再卷积，VALID不能补0  
    # 返回 Tensor  
    ############################################################################################  
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
  
############################################################################################  
# tf.placeholder 形参定义  
# 参数 dtype 数据类型  
# 参数 shape 形状，默认是None，  
############################################################################################  
x = tf.placeholder(tf.float32, [None, 784])  
y_ = tf.placeholder(tf.float32, [None, 10])  
############################################################################################  
# tf.reshape 重定形状  
# 参数 tensor 输入数据  
# 参数 shape 形状                按此shape生成相应数组，但-1是特例，表示有此维度，但是数值不定  
# 返回 Tensor  
############################################################################################  
x_image = tf.reshape(x, [-1, 28, 28, 1])  
  
w_conv1 = weight_variable([5, 5, 1, 32])  
b_conv1 = bias_variable([32])  
############################################################################################  
# tf.nn.relu RELU函数  = max(0,features)  
# 参数 features 输入特征Tensor  
# 返回 Tensor  
############################################################################################  
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  
  
w_conv2 = weight_variable([5, 5, 32, 64])  
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  
  
w_fc1 = weight_variable([7 * 7 * 64, 1024])  
b_fc1 = bias_variable([1024])  
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)  
  
keep_prob = tf.placeholder(tf.float32)  
############################################################################################  
# tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)   防止过拟合   在对输入的数据进行一定的取舍，从而降低过拟合  
# 参数 x 输入数据  
# 参数 keep_prob 保留率             对输入数据保留完整返回的概率  
# 返回 Tensor  
############################################################################################  
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)  
  
w_fc2 = weight_variable([1024, 10])  
b_fc2 = bias_variable([10])  
############################################################################################  
# tf.matmul(a, b): 矩阵相乘  a * b  
# 参数 a 矩阵Tensor  
# 参数 b 矩阵Tensor  
# 返回 Tensor  
############################################################################################  
############################################################################################  
# tf.nn.softmax(logits, dim=-1, name=None): SoftMax函数  softmax = exp(logits) / reduce_sum(exp(logits), dim)  
# 参数 logits 输入            一般输入是logit函数的结果  
# 参数 dim 卷积核             指定是第几个维度，默认是-1，表示最后一个维度  
# 返回 Tensor  
############################################################################################  
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)  
  
############################################################################################  
# tf.nn.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None) 求和函数  
# 参数 input_tensor 输入数据             可以是值，也可以是多维矩阵  
# 参数 axis 求和方式                     默认是全求和；如果是0，就是按列求和；如果1，就是按行求和  
# 参数 keep_dims 是否保留原有维度样式    True表示是，False表示不是  
# 返回 Tensor  
############################################################################################  
############################################################################################  
# tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)  均值函数  
# 参数 input_tensor 输入数据             可以是值，也可以是多维矩阵  
# 参数 axis 求和方式                     默认是全求和；如果是0，就是按列求和；如果1，就是按行求和  
# 参数 keep_dims 是否保留原有维度样式    True表示是，False表示不是  
# 返回 Tensor  
############################################################################################  
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))  
############################################################################################  
# tf.train.AdamOptimizer Adam优化算法  
# __init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')  
# 返回 Optimizer  
# 是一个寻找全局最优点的优化算法，引入了二次方梯度校正。相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快  
############################################################################################  
############################################################################################  
# tf.train.Optimizer.minimize 优化算法之最小化函数 主要参数是loss  
# 参数 loss 损失量  
# 返回 Operation  
# minimize需要run唤起  
############################################################################################  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  
############################################################################################  
# tf.argmax(input, axis=None, name=None, dimension=None) 对矩阵按行或列进行最大值下标提取  
# 参数 input 输入  
# 参数 axis axis轴         0表示按列，1表示按行  
# 参数 name 名称  
# 参数 dimension维度       和axis功能一样，默认axis取值优先。新加的字段  
# 返回 Tensor  
############################################################################################  
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  
############################################################################################  
# tf.cast(x, dtype, name=None) 类型转换  
# 参数 x 输入  
# 参数 dtype 转换后的类型  
# 返回 Tensor  
############################################################################################  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
  
############################################################################################  
# tf.global_variables_initializer 初始化所有全局变量  
# 返回 Operation  
############################################################################################  
tf.global_variables_initializer().run()       #启动Session  
for i in range(20000):  
    ############################################################################################  
    # Datasets.train.next_batch 批量处理记录数  
    # 返回 [image,label]  
    ############################################################################################  
    batch = mnist.train.next_batch(50)  
    if i%100 == 0:  
        ############################################################################################  
        # Tensor.eval Tensor的执行函数，只能在session启动后面执行  
        # 参数 feed_dict  形参列表  
        # 返回 Tensor  
        ############################################################################################  
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})  
        print("step %d, training accuracy %g" %(i, train_accuracy))  
    ############################################################################################  
    # Operation.run 执行函数 == tf.get_default_session().run(op).  
    # 参数 feed_dict  形参列表  
    ############################################################################################  
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob: 0.5})  
  
print("test accuracy %g" %accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))  