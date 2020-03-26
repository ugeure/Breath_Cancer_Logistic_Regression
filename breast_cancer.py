# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:12:27 2020

@author: ugeure
"""
#importing libs
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
#import the data

df=pd.read_csv("Breast_cancer_data.csv")
    #getting data informations
#df.info()
    #input and output selection
y=df.diagnosis.values #converting numpy array
x_data = df.drop(["diagnosis"],axis=1) #input selection
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) #normalization input data
#%% train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
    #Transpose of data_vectors #because of the calculations 
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
#%% initialize weights and bias functian
def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.001
    return w,b
#%% sigmoid function
def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
#%%  forward and backward propogation

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1] is for avarage
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for avarage
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for avarage
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients
#%% update parameters
    #in order to decrease cost we must update weights and bias 
def update(w, b, x_train, y_train, learning_rate,num_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is num_iteration times
    for i in range(num_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 12 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration {}: {}" .format(i, cost))
#        if cost<0.28 : #you can set the cost value to break the iterations
#                break # this setting is useful for over-fitting
    new_w_b = {"weight": w,"bias": b} 
    return new_w_b, gradients, cost_list
#%% predict
    
def prediction(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
    for j in range(z.shape[1]):
        if z[0,j]<= 0.5:
            y_prediction[0,j] = 0
        else:
            y_prediction[0,j] = 1

    return y_prediction
#%% Logistic Regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize functions to getting  values
    dimension =  x_train.shape[0]  #
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = prediction(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
 
# we can use the final function to getting our score
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1.5, num_iterations = 300) 


            
            