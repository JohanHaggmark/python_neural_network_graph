from time import sleep
import time as t
import numpy as np
#from graphicNeuralNetwork import *
import matplotlib.pyplot as plt
 
import matplotlib.animation as animation
from matplotlib import style
   

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def animate(weights_layer11, weights_layer21):
    
    plt.plot([1,2],[3,1],linewidth= (weights_layer11[[0],[0]] + 1)*5)
    plt.plot([1,2],[3,3],linewidth= (weights_layer11[[0],[1]] + 1)*5)
    plt.plot([1,2],[3,5],linewidth= (weights_layer11[[0],[2]] + 1)*5)
    plt.plot([1,2],[3,7],linewidth= (weights_layer11[[0],[3]] + 1)*5)

    plt.plot([1,2],[5,1],linewidth= (weights_layer11[[1],[0]] + 1)*5)
    plt.plot([1,2],[5,3],linewidth= (weights_layer11[[1],[1]] + 1)*5)
    plt.plot([1,2],[5,5],linewidth= (weights_layer11[[1],[2]] + 1)*5)
    plt.plot([1,2],[5,7],linewidth= (weights_layer11[[1],[3]] + 1)*5)


    plt.plot([2,3],[1,3],linewidth= (weights_layer21[[0],[0]] + 1)*5)
    plt.plot([2,3],[1,5],linewidth= (weights_layer21[[0],[1]] + 1)*5)

    plt.plot([2,3],[3,3],linewidth= (weights_layer21[[1],[0]] + 1)*5)
    plt.plot([2,3],[3,5],linewidth= (weights_layer21[[1],[1]] + 1)*5)

    plt.plot([2,3],[5,3],linewidth= (weights_layer21[[2],[0]] + 1)*5)
    plt.plot([2,3],[5,5],linewidth= (weights_layer21[[2],[1]] + 1)*5)
    
    plt.plot([2,3],[7,3],linewidth= (weights_layer21[[3],[0]] + 1)*5)
    plt.plot([2,3],[7,5],linewidth= (weights_layer21[[3],[1]] + 1)*5)
  
   # def runAnimation(weights_layer11, weights_layer21):
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidprime(x):
    return x*(1-x)
      #,[1,1],[0,0],[0,0],[1,0],[0,0],[0,0],[1,1],[1,1]
    #,[25,37],[2,3],[1,2],[35,6],[3,3],[5,1],[30,30],[29,32]
#network test
facit = np.array([[1,0],[1,0],[0,1],[0,1],[1,1],[0,0],[0,0],[1,0],[0,0],[0,0],[1,1],[1,1]])
#this is the input data
inputdata_layer1 = np.array([[30,11],[26,7],[2,25],[8,15],[25,37],[2,3],[1,2],[35,6],[3,3],[5,1],[30,30],[29,32]
])

#weights for every synapse is created
def runWeights():
    weights_layer1 = 2*np.random.random((2,4))-1
    weights_layer2 = 2*np.random.random((4,2))-1
    return weights_layer1, weights_layer2

weights_layer1, weights_layer2 = runWeights()

animate(weights_layer1, weights_layer2)

ani = animation.FuncAnimation(fig, animate(weights_layer1, weights_layer2),frames=100, interval=100)

#Animations.runAnimation(weights_layer1, weights_layer2)
for j in range(1000):
      #  buildWeights(weights_layer1, weights_layer2)
      #  print("error: " + str(np.mean(np.abs(errorcheck_layer3))))
    plt.clf()
   
    animate(weights_layer1, weights_layer2)
     #forward####
    hidden_layer2 =  sigmoid(np.dot(inputdata_layer1, weights_layer1))
    output_layer3 = sigmoid(np.dot(hidden_layer2, weights_layer2))
    #############

    
    errorcheck_layer3 = facit - output_layer3
    delta_layer3 = errorcheck_layer3 * sigmoidprime(output_layer3)
    errorcheck_layer2 = delta_layer3.dot(weights_layer2.T)
    
    delta_layer2 = errorcheck_layer2 * sigmoidprime(hidden_layer2)
      
    weights_layer2 += hidden_layer2.T.dot(delta_layer3)
    weights_layer1 += inputdata_layer1.T.dot(delta_layer2)   
        
    plt.show()
    #plt.pause(0.001)
    #t.sleep(0.001)
    plt.pause(0.01)
   
       
print("error: " + str(np.mean(np.abs(errorcheck_layer3))))
print("output for this trash: ")
print(output_layer3)
#buildWeights(weights_layer1, weights_layer2)

#https://www.youtube.com/watch?v=h3l4qz76JhQ