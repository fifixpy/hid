# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

dir='D:\data_set\mnist'

# 导入数据
mnist = input_data.read_data_sets(dir, one_hot=True)
    #read_data_sets 检查目录下有没有想要的数据，没有的话下载，然后进行解压，返回一个Datasets包含train, validation, test


#Print the shape of mist
print (mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.train.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)


#建立一个拥有一个线性层的softmax回归模型

# 定义模型
# y=wx+b #权重值W和偏置量b
x = tf.placeholder(tf.float32, [None, 784])
    #神经网路的输入：输入任意数量的MNIST图像, 每一张图展平成28*28维的向量（2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]）
    #None表示此张量的第一个维度可以是任何长度的，也就是图片数量不定的意思
W = tf.Variable(tf.zeros([784, 10]))# 一个Variable代表一个可修改的张量
b = tf.Variable(tf.zeros([10]))
    #W是一个784*10的矩阵，有784个特征和10个输出值
    #b是一个10维的向量，有10个分类，所以我们可以直接把它加到输出上面y = tf.matmul(x, W) + b
y = tf.matmul(x, W) + b


# 设置优化方法（损耗和优化器）
y_ = tf.placeholder(tf.float32, [None, 10])#添加一个新的占位符用于输入正确值，代表对应某一图片的类别

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))#计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #TensorFlow用梯度下降算法以0.01的学习速率最小化交叉熵
    # 梯度下降算法是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动

# 初始化模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# 训练100000次，每次选择100个作为输入
for i in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(100)#每一步迭代，我们都会加载100个训练样本，然后执行一次train_step
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})#feed_dict将x 和 y_张量占位符用训练训练数据替代
    if(i%10000==0):
        print(i)


# 测试训练模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #tf.argmax给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
    #比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))