from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 权重初始化

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# 卷积

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # x 是一个 4 维张量，shape为[batch,height,width,channels]
    #卷积核移动步长为1，填充类型为SAME,可以不丢弃任何像素点


# 池化

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    #采用最大池化，也就是取窗口中的最大值作为结果
    #x 是一个4维张量，shape为[batch,height,width,channels]
    #ksize表示pool窗口大小为2x2,也就是高2，宽2
    #strides，表示在height和width维度上的步长都为2


# MNIST数据存放的路径
file = "D:\data_set\mnist"

# 导入数据
mnist = input_data.read_data_sets(file, one_hot=True)

# 模型的输入和输出
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



# 第一层卷积（卷积+max_pooling）

W_conv1 = weight_variable([5, 5, 1, 32])# 初始化 W 为[5,5,1,32]的张量，表示卷积核大小为 5*5，第一层网络的输入通道数为1 ，输出通道个数分别为 32
b_conv1 = bias_variable([32])# 初始化b为[32],即输出大小
x_image = tf.reshape(x, [-1,28,28,1])

    # 把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1],灰度图通道数为1
    #-1表示自动推测这个维度的size

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

    #把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
    #h_pool1的输出即为第一层网络输出，shape为[batch,14,14,32]



# 第二层卷积

W_conv2 = weight_variable([5, 5, 32, 64])# 卷积核大小依然是5*5，这层的的输入通道数为32 ，输出通道个数分别为 64
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #h_pool2即为第二层网络输出，shape为[batch,7,7,64]



# 密集连接层

W_fc1 = weight_variable([7 * 7 * 64, 1024])#加入一个有1024个神经元的全连接层；W 的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#把池化层输出的张量reshape成[batch, 7*7*64]的张量，乘上权重矩阵，加上偏置，然后对其使用ReLU
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



# Dropout

keep_prob = tf.placeholder("float")#为了减少过拟合，在输出层前加入dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)




# 输出层

    #最后，添加一个softmax层
    #可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



# 创建Session
sess = tf.InteractiveSession()
# 初始化权重变量
sess.run(tf.global_variables_initializer())



# 训练和评估模型

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    #预测值和真实值之间的交叉墒

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #train op, 使用ADAM优化器来做梯度下降。学习率为0.0001

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。
    #因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #计算正确预测项的比例，因为tf.equal返回的是布尔值，
    #使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值

sess.run(tf.global_variables_initializer())#初始化变量

##开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))