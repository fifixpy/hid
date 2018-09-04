import cifar10
import cifar10_input
import tensorflow as tf
import os
import tarfile

#解压缩
filepath = 'D:/data_set/cifar10_data/cifar-10-binary.tar.gz'
dest_directory = 'D:/data_set/cifar10_data'
extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
if not os.path.exists(extracted_dir_path):
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

data_dir = 'D:/data_set/cifar10_data/cifar-10-batches-bin'
batch_size = 100

#生成CIFAR-10的训练数据和训练标签数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

#生成CIFAR-10的测试数据和测试标签数据
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

sess = tf.InteractiveSession()
tf.global_variables_initializer()
tf.train.start_queue_runners()
print(images_train)
print(images_test)