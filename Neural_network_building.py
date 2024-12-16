## In this script is a pretrained artificial neural network from Building AI ##
# The network predicts the price of a cabin based it's attributes.
# I solved the task to fix it at some points and to use it to make a prediction for a cabin-price.
# The network consists of an input layer with five nodes, a hidden layer with two nodes, a second hidden layer with two nodes, 
# and finally an output layer with a single node. In addition there is a single bias node for each hidden layer and the output layer.
# Although it was not the hardest task I like it, because it gives good insights of the functioning of a neural network.



# Import of necessary libraries.

import numpy as np



# The pretrained weights of the neural network for different layers.

w0 = np.array([[ 1.19627687e+01,  2.60163283e-01],
               [ 4.48832507e-01,  4.00666119e-01],
                              [-2.75768443e-01,  3.43724167e-01],
                   [ 2.29138536e+01,  3.91783025e-01],
                   [-1.22397711e-02, -1.03029800e+00]])

w1 = np.array([[11.5631751 , 11.87043684],
                   [-0.85735419,  0.27114237]])

w2 = np.array([[11.04122165],
                   [10.44637262]])



# The pretrained bias-nodes of the neural network for different layers.

b0 = np.array([-4.21310294, -0.52664488])
b1 = np.array([-4.84067881, -4.53335139])
b2 = np.array([-7.52942418])



# The different activation functions used.

def hidden_activation(z):

    z[z <= 0] = 0   # The ReLU activation function was missing.
            
    return z

def output_activation(z):
    # The identity (linear) activation function was missing.
    return z




# The test attributes of a cabin.

x_test = [[82, 2, 65, 3, 516]]




# The forward pass algorithm.
# The forward pass is running the input variables through the neural network to obtain output, in this case the price for a cabin of given attributes.
# Inside the for loop the functioning of in the neural network is nicely shown. For the layers the linear combination is calculated first. Than the bias-nodes are added. 
# The result is used as the input for an activation function.


for item in x_test:
    h1_in = np.dot(item, w0)            # this calculates the linear combination of inputs and weights. It was missing the bias term.
    h1_in = h1_in+b0
    h1_out = hidden_activation(h1_in)   # apply of ReLU activation function.
    
    h2_in = np.dot(h1_out, w1)          # the output of the previous layer is the input for this layer. It was missing the bias term.
    h2_in = h2_in+b1
    h2_out = hidden_activation(h2_in)   # apply of ReLU activation function.
    
    out_in = np.dot(h2_out, w2)         # the output of the previous layer is the input for this layer. It was missing the bias term.
    out_in = out_in + b2
    out = output_activation(out_in)     # apply of identity activation function.
    print(out)                          # print of the output (price-prediction).


