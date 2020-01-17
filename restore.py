#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:51:56 2020

@author: afaanansari
"""


import matplotlib.pyplot as plt
import tensorflow as ta
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
tf=ta.compat.v1
tf.disable_eager_execution()

def read_dataset():
    df=pd.read_csv("//Users//afaanansari//Desktop//spyder//fake_notes//note.csv")
    X=df[df.columns[0:4]].values
    y1=df[df.columns[4]]
    
    encoder = LabelEncoder()
    encoder.fit(y1)
    y=encoder.transform(y1)
    Y=one_hot_encode(y)
    #print(X.shape)
    return(X,Y,y1)

def one_hot_encode(labels):
    n_labels=len(labels)
    n_unique_labels=len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return one_hot_encode

X,Y,y1 = read_dataset()

model_path = "//Users//afaanansari//Desktop//spyder//Fakenote-0"
learning_rate = 0.3
training_epochs =100
cost_history = np.empty(shape=[1],dtype=float)
n_dim=4
n_class = 2 

n_hidden_1 = 10
n_hidden_2 = 10
n_hidden_3 = 10
n_hidden_4 = 10


x = tf.placeholder(tf.float32,[None,n_dim])
W = ta.Variable(tf.zeros([n_dim,n_class]))
b = ta.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None,n_class])

def multilayer_perceptron(x , weights , biases):
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    out_layer = tf.matmul(layer_4,weights['out']) + biases['out']
    return out_layer

weights = {
    
    'h1' : ta.Variable(tf.truncated_normal([n_dim , n_hidden_1])),
    'h2' : ta.Variable(tf.truncated_normal([n_hidden_1 , n_hidden_2])),
    'h3' : ta.Variable(tf.truncated_normal([n_hidden_2 , n_hidden_3])),
    'h4' : ta.Variable(tf.truncated_normal([n_hidden_3 , n_hidden_4])),
    'out' : ta.Variable(tf.truncated_normal([n_hidden_4 , n_class])),
    }

biases = {
    
    'b1' : ta.Variable(tf.truncated_normal([n_hidden_1])),
    'b2' : ta.Variable(tf.truncated_normal([n_hidden_2])),
    'b3' : ta.Variable(tf.truncated_normal([n_hidden_3])),
    'b4' : ta.Variable(tf.truncated_normal([n_hidden_4])),
    'out' : ta.Variable(tf.truncated_normal([n_class])),
    }

inti = tf.global_variables_initializer()



y=multilayer_perceptron(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y , labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()

sess.run(inti)

saver = tf.train.Saver()
saver = tf.train.import_meta_graph("/Users/afaanansari/Desktop/spyder/model.ckpt-100.meta")

saver.restore(sess,"/Users/afaanansari/Desktop/spyder/model.ckpt-100")


prediction = tf.argmax(y,1)
correct_prediction = tf.equal(prediction , tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print('************************************************************************')
print("0 stands for Fake Note & 1 stands for Real Note")
print('************************************************************************')

for i in range(754 , 768) :
    predicition_run = sess.run(prediction,feed_dict = {x : X[i].reshape(1,4)})
    accuracy_run = sess.run(accuracy ,feed_dict = {x: X[i].reshape(1,4) , y_ : X[i].reshape(1,4)  })
    print("orignal : ", y1[i],"predicted : ",predicition_run)


