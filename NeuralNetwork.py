# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:02:50 2018

@author: canon
"""
import numpy as np
import matplotlib.pyplot as plt

class Network(object):
    
    #input_nodes
   # hidden_nodes
   # output_nodes
  #  hidden_layers
    def __init__(self,input_nodes, hidden_nodes, output_nodes, hidden_layers):
         self.input_nodes = input_nodes
         self.hidden_nodes = hidden_nodes
         self.output_nodes = output_nodes
         self.hidden_layers = hidden_layers
         self.setWeightsInput(input_nodes, hidden_nodes)
         self.setWeightsHidden(hidden_nodes, output_nodes, hidden_layers)
    
    def setWeightsInput(self, input_nodes, hidden_nodes):
        self.weights_to_hidden_layer_1 = 2 * np.random.random((input_nodes,hidden_nodes))
      

    def setWeightsHidden(self, hidden_nodes, output_nodes, hidden_layers):     
        self.weights_in_hidden_layer = 2* np.random.random((hidden_layers-1, hidden_nodes, hidden_nodes))
        self.weights_to_output_layer = 2* np.random.random((hidden_nodes, output_nodes))
        
    def feedForward(self, input_data):
        self.input_data = input_data
        if(input_data.ndim > 1):
            self.training_nodes(input_data)
        else:
            self.forward()
            
    def forward(self):
        return
   
    def training_nodes(self, input_data):
       # self.layer = np.empty((self.hidden_layers, self.hidden_nodes))
        self.layer = np.empty((len(input_data), self.hidden_layers, self.hidden_nodes))
        self.layer[:, 0, :] = self.sigmoid(np.dot(input_data, self.weights_to_hidden_layer_1))
    
        for i in range(1, self.hidden_layers):
            self.layer[:, i, :] = self.sigmoid(np.dot(self.layer[:,i-1,:], self.weights_in_hidden_layer[i-1,:,:]))
   
            
        self.output_layer = self.sigmoid(np.dot(self.layer[:,-1,:], self.weights_to_output_layer))
        return self.output_layer    
        
    def train(self, input_data, output_layer_facit, row):
        for i in range(row):    
            self.feedForward(input_data)
            self.backpropagate(input_data, output_layer_facit)
            self.mean_error()
        
    def train_relu(self, input_data, output_layer_facit, row, learning_rate):
        for i in range(row):    
            self.forward_relu(input_data)
            self.backpropagate_relu(input_data, output_layer_facit, learning_rate/np.size(input_data, 0))
            self.mean_error()
            
    def backpropagate(self, input_data, output_layer_facit):
        
       
        #check error for predicted and facit
        self.error = output_layer_facit - self.output_layer
        #backpropagate
        # reluvariable = relu_derivate(output_layer_predicted)             
        delta = self.error * self.sigmoid_prime(self.output_layer)         
        #must store all hidden_layer when feedforward
       
        self.error = np.dot(delta, self.weights_to_output_layer.T)
        self.weights_to_output_layer += 1 * np.dot(self.layer[:,self.hidden_layers-1,:].T, delta)  
        for i in reversed(range(self.hidden_layers-1)):
            #error = np.dot(delta, self.weights_in_hidden_layer[i,:,:].T)
            delta = self.error * self.sigmoid_prime(self.layer[:,i+1,:])
            self.error = np.dot(delta, self.weights_in_hidden_layer[i,:,:].T)
            
            self.weights_in_hidden_layer[i,:,:] += 1* np.dot(self.layer[:,i,:].T, delta)
          
       # delta = error * self.sigmoid_prime(self.layer[:,0,:])
        #error = np.dot(delta, self.weights_to_hidden_layer_1.T)
        delta = self.error * self.sigmoid_prime(self.layer[:,0,:])
        #delta = error * self.sigmoid_prime(self.layer[:,0,:])
        self.weights_to_hidden_layer_1 += 1* np.dot(input_data.T, delta)

      #   error = output_layer_facit - output_layer_predicted
      #   delta_output_layer = error * relu_derivate(output_layer_predicted)
      
      #  error_hidden_layer_2 = np.dot(delta_output_layer, weights_to_output_layer.T)
      #  delta_hidden_layer_2 = error_hidden_layer_2 * sigmoid_prime(hidden_layer_2)
    
      #  error_hidden_layer_1 = np.dot(delta_hidden_layer_2, weights_to_hidden_layer_2)
      #  delta_hidden_layer_1 = error_hidden_layer_1 * sigmoid_prime(hidden_layer_1)
    

        
       # weights_to_output_layer += 0.01 * np.dot(hidden_layer.T, delta_output_layer)
       # weights_to_hidden_layer_2 += 2 * np.dot(hidden_layer_1.T, delta_hidden_layer_2)

       # weights_to_hidden_layer_1 += 2 * np.dot(input_layer_training_nodes.T, delta_hidden_layer_1)
       
    def forward_relu(self, input_data):
        self.training_nodes_relu(input_data)
        
    def backpropagate_relu(self, input_data, output_layer_facit, learning_rate):
        self.error = output_layer_facit - self.output_layer      
        delta = self.error * self.relu_derivate(self.output_layer)         
    
        self.error = np.dot(delta, self.weights_to_output_layer.T)
        self.weights_to_output_layer += learning_rate * np.dot(self.layer[:,self.hidden_layers-1,:].T, delta)  
        for i in reversed(range(self.hidden_layers-1)):
            #error = np.dot(delta, self.weights_in_hidden_layer[i,:,:].T)
            delta = self.error * self.relu_derivate(self.layer[:,i+1,:])
            self.error = np.dot(delta, self.weights_in_hidden_layer[i,:,:].T)
            
            self.weights_in_hidden_layer[i,:,:] += learning_rate* np.dot(self.layer[:,i,:].T, delta)
          
        delta = self.error * self.relu_derivate(self.layer[:,0,:])
        self.weights_to_hidden_layer_1 += learning_rate* np.dot(input_data.T, delta)
       
    def training_nodes_relu(self, input_data):
       # self.layer = np.empty((self.hidden_layers, self.hidden_nodes))
        self.layer = np.empty((len(input_data), self.hidden_layers, self.hidden_nodes))
        self.layer[:, 0, :] = self.relu(np.dot(input_data, self.weights_to_hidden_layer_1))
    
        for i in range(1, self.hidden_layers):
            self.layer[:, i, :] = self.relu(np.dot(self.layer[:,i-1,:], self.weights_in_hidden_layer[i-1,:,:]))
            
        self.output_layer = self.sigmoid(np.dot(self.layer[:,-1,:], self.weights_to_output_layer))
        return self.output_layer    
    
    def mean_error(self):
        print(str(np.mean(np.abs(self.error))))
        
    def sigmoid(self,x):
        x = 1/(1+np.exp(-x))
        return x

    def sigmoid_prime(self,x):
        return x*(1-x)

    def relu(self,x):
        x = np.maximum(0, x)
        return x

    def relu_derivate(self,x):
        x[x<0] = 0
        x[x>0] = 1
        return x
    
    
    def animate(self):
        plt.clf()
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes):
                plt.plot([1,2], [i, j], linewidth= (self.weights_to_hidden_layer_1[[i],[j]] + 1)*5)
        
        for i in range(self.hidden_layers-1):
            for j in range(self.hidden_nodes):
                for h in range(self.hidden_nodes):    
                    plt.plot([i+2,i+3],[j,h],linewidth= (self.weights_in_hidden_layer[[i],[j],[h]] + 1)*5)
            
        for i in range(self.hidden_nodes):
            for j in range(self.output_nodes):
                plt.plot([self.hidden_layers+1, self.hidden_layers+2], [i, j], linewidth= (self.weights_to_output_layer[i,j] + 1) * 5)
        plt.show()      
        plt.pause(0.1)

    