# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:29:42 2018

@author: Johan
"""

import numpy as np

#import simple_neural_network_graph
import NeuralNetwork as nn


#example of data
input_layer_training_nodes = np.array([[5,5,4,0,1,0],[4,3,6,1,1,0],[2,5,6,0,0,1],[4,8,2,1,0,1],[6,4,2,1,0,0],[4,1,6,1,0,1],[3,3,5,1,0,0],
                                       [1,3,0,5,3,2],[1,3,0,7,5,2],[0,1,0,4,2,7],[0,1,0,6,4,2],[1,3,0,6,6,1],[0,0,0,5,6,4],[0,0,1,4,5,4],
                                       [5,4,5,5,4,2],[4,5,0,7,5,4],[7,1,4,4,2,7],[6,6,1,6,4,4],[4,3,4,6,6,1],[1,4,5,5,6,4],[3,3,3,4,5,4],
                                      ])
            
output_layer_facit = np.array([[1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0],
                               [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 
                               [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], 
                              ])

network = nn.Network(6,6,2,2)
##################################################################################################




#example of data
input_layer_data = np.array([[8,0,0],[8,0,0],[0,0,6], [0,0,7]])
output_layer_data = np.array([[1,0],[1,0],[0,1], [0,1]])

input_layer_data = np.array([[6,0,0],[8,0,0],[5,0,0], [7,0,0]])
output_layer_data = np.array([[1,0],[1,0],[0,0], [1,0]])

network = nn.Network(3,6,2,4)
##########################################################################



#training with sigmoid
for i in range(100):   
    network.train(input_layer_training_nodes, output_layer_facit, 1)
    network.animate()
    
    
#training with relu
for i in range(50):   
    network.train_relu(input_layer_training_nodes, output_layer_facit, 1, 0.01)
    network.animate()
    
