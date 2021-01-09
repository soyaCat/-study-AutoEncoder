import numpy as np
import random
import datetime
import time
from collections import deque
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import os

mnist = input_data.read_data_sets("./mnist/data/", one_hot= True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
train_x = np.reshape(train_x,(-1,28,28,1))

learning_rate = (1e-5)*5
batch_size = 256
train_epochs = 2000
test_epochs = 100
generate_switch = False

train_mode = True
load_model = False
generate_image_from_point = True
# model save and load
game = "AutoEncoder"
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "D:/program/python/saved_models/" + game + "/" + date_time + "_AutoEncoder"
load_path = "D:/program/python/saved_models/" + game + "/"

print_interval = 100
save_interval = 500


class AEModel:
    def __init__(self):
        self.input_EN = tf.placeholder(shape = [None, 28, 28, 1], dtype = tf.float32)
        self.input_DC = tf.placeholder(shape = [None, 2], dtype = tf.float32)

        self.conv1 = tf.layers.conv2d(
            inputs = self.input_EN,
            filters = 32,
            activation= tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[1,1],
            padding="SAME",
        )
        self.conv2 = tf.layers.conv2d(
            inputs = self.conv1,
            filters = 64,
            activation= tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[2,2],
            padding="SAME",
        )
        self.conv3 = tf.layers.conv2d(
            inputs = self.conv2,
            filters = 64,
            activation= tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[2,2],
            padding="SAME",
        )
        self.conv4 = tf.layers.conv2d(
            inputs = self.conv3,
            filters = 64,
            activation= tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[2,2],
            padding="SAME",
        )
        self.flat_EN = tf.layers.flatten(self.conv4)
        self.out_EN = tf.layers.dense(self.flat_EN, 2, activation=None)


        self.flat = tf.layers.dense(self.out_EN, 3136, activation=None)
        if generate_switch:
            self.flat = tf.layers.dense(self.input_DC, 3136, activation= None)
        
        self.reshape = tf.reshape(self.flat, shape = [-1,7,7,64])
        self.conv_t1 = tf.layers.conv2d_transpose(
            inputs = self.reshape,
            filters = 64,
            activation = tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[1,1],
            padding = "SAME",
        )
        self.conv_t2 = tf.layers.conv2d_transpose(
            inputs = self.conv_t1,
            filters = 64,
            activation = tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[2,2],
            padding = "SAME",
        )
        self.conv_t3 = tf.layers.conv2d_transpose(
            inputs = self.conv_t2,
            filters = 32,
            activation = tf.nn.leaky_relu,
            kernel_size = [3,3],
            strides=[2,2],
            padding = "SAME",
        )
        self.out = tf.layers.conv2d_transpose(
            inputs = self.conv_t3,
            filters = 1,
            activation = tf.nn.sigmoid,
            kernel_size = [3,3],
            strides=[1,1],
            padding = "SAME",
        )
        self.answer = tf.placeholder(shape = [None, 28, 28, 1], dtype = tf.float32)
        self.loss = tf.losses.mean_squared_error(self.out, self.answer)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class AutoEncoder:
    def __init__(self):
        self.Model = AEModel()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        
        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    #The Image compressed by Encoder, and Decompressed by Decoder
    def run_AE(self, data):
        generate_switch = False
        generated_image = self.sess.run(self.Model.out, feed_dict = {self.Model.input_EN: data})
        return generated_image

    #The pointData decompressed by Decoder, 
    #We can expect Decoder generate random number image from pointData
    def run_DC(self, data):
        generate_switch = True
        dump = np.random.randn(1,28,28,1)
        generated_image = self.sess.run(self.Model.out, feed_dict = {self.Model.input_EN: dump,
                                                                    self.Model.input_DC: data})
        return generated_image

    def train_AE(self, data):
        generate_switch = False
        self.mini_batch = random.sample(data, batch_size)
        _, loss = self.sess.run([self.Model.UpdateModel, self.Model.loss], feed_dict = {self.Model.input_EN: self.mini_batch,
                                                                                        self.Model.answer: self.mini_batch})
        return loss
    
    def show2Image(self, image1, image2):
        image1 = np.uint8(255*np.reshape(image1,(28,28)))
        image2 = np.uint8(255*np.reshape(image2,(28,28)))
        cv2.imshow("image1",image1)
        cv2.imshow("image2", image2)

    def show1Image(self, image1):
        image = np.uint8(255*np.reshape(image1,(28,28)))
        cv2.imshow("image",image)

    def Make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        Summary = tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()
        return Summary, Merge

    def Write_Summray(self, loss, epoch):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={self.summary_loss: loss}), epoch)

    def save_model(self, epoch):
        path = save_path + "/model/ep{}/".format(epoch)
        os.makedirs(path)
        self.Saver.save(self.sess, path + "model")
         


if __name__ == "__main__":
    data = []
    for i in range(len(train_x)):
        data.append(train_x[i])
    AE = AutoEncoder()
    losses = []
    for epoch in range(train_epochs+test_epochs):
        if epoch > train_epochs:
            cv2.destroyAllWindows()
            print("\n")
            print("train_finish: turn off train_mode")
            print("press any key if you want to show another image")
            train_mode = False
        
        if train_mode:
            loss = AE.train_AE(data)
            losses.append(loss)
        
        else:
            if generate_image_from_point == False:
                print("show image encode and decode....")
                images = []
                images.append(data[random.randint(0,len(data))])
                generated_image = AE.run_AE(images)
                AE.show2Image(images[0], generated_image)
                ret=cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            else:
                print("generate image from point...")
                arr = np.random.randn(1,2)*10
                print(arr)
                generated_image = AE.run_DC(arr)
                AE.show1Image(generated_image)
                ret = cv2.waitKey(0)
                cv2.destroyAllWindows()

        
        if epoch % print_interval == 0 and train_mode:
            print("epoch:{}/ loss:{}".format(epoch, np.mean(losses)))
            AE.Write_Summray(np.mean(losses),epoch)
            losses = []
            cv2.destroyAllWindows()
            images = []
            images.append(data[random.randint(0,len(data))])
            generated_image = AE.run_AE(images)
            AE.show2Image(images[0], generated_image)
            ret = cv2.waitKey(1)

        if epoch % save_interval == 0 and epoch != 0:
            AE.save_model(epoch)
            print("Save Model {}".format(epoch))






        

        
