import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Initialization: 
#1. Network has one hidden layer with 25 nodes, one input layer with 784 input nodes, one output layer with 4 output nodes

def init_params():
    #Initialize the weights of the first and second layer, as well the bias with random small values
    w1 = np.random.rand(25, 784) - 0.5
    b1 = np.random.rand(25, 1) - 0.5
    w2 = np.random.rand(4, 25) - 0.5

    return w1, b1, w2

# Definition of the ReLu Activation Function
def relu(z):
    return np.maximum(0, z)

#Definition of the derivative of ReLu Activation Function
def relu_deri(z):
    return z > 0

#Definition of the Softmax Activation Function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

#Definition of the derivative of Softmax Activation Function
def softmax_deri(z):
    return softmax(z) * (1.0 - softmax(z))

#Definition of the Sigmoid Activation Function
def _sigmoid(z):
    return 1/(1+np.exp(-z))

#Definition of the derivative of Sigmoid Activation Function
def _sigmoid_deri(z):
    return _sigmoid(z) * (1.0 - _sigmoid(z))

#Forward Propagation:
def forward_prop(w1, b1, w2, x):    
    net1 = w1.dot(x) + b1           #Calculate the dot product of the input with the first layer weights
    a1 = relu(net1)                 #Pass it through the ReLu Activation Function
    
    net2 = w2.dot(a1)               #Calculate the dot product of the hidden layer output with its weights
    a2 = softmax(net2)              #Pass it through the Softmax Activation Function
    
    return net1, a1, net2, a2

#Back Propagation:
def back_prop(net1, a1, net2, a2, w1, w2, x, y,m):
    
    dnet2 = a2 - y                                      #The error is the actual output - target 
    dw2 = (1/m) * dnet2.dot(a1.T)                       #Backpropagating from output layer to hidden layer
    
    dnet1 = w2.T.dot(dnet2) * relu_deri(net1)           #Taking derivations of activations at hidden layer
    dw1 = (1/m) * dnet1.dot(x.T)                        #Taking Derivative with respect to first layer weights
    db1 = (1/m) * np.sum(dnet1, 1)                      #Taking derivatives of bias term
    
    return dw1, db1, dw2

def update_params(w1, b1, w2, dw1, db1, dw2, learning_rate):
    w1 = w1 - learning_rate * dw1                       #Getting updated weights for first layer
#     b1 = b1 - learning_rate * db1
    b1 -= learning_rate * np.reshape(db1, (25,1))       #Getting updated bias value
    w2 = w2 - learning_rate * dw2                       #Getting updated weights for second layer 
    
    return w1, b1, w2


#One hot encoding:
def one_hot_encoding(max_idx):

        ip_arr = np.array(max_idx)
        op_arr = np.zeros((1,4))
        op_arr[np.arange(ip_arr.size), ip_arr] = 1

        return op_arr[0]

#Function for retreiving accuracy of model   
def accuracy(y_true, y_pred):
    if not (len(y_true) == len(y_pred)):
        print('Size of predicted and true labels not equal.')
        return 0.0
    corr = 0
    for i in range(0,len(y_true)):
        corr += 1 if (y_true[i] == y_pred[i]).all() else 0

    return corr/len(y_true)

#Gradient Descent:
def grad_desc(x, y, itera, learning_rate,m):
    w1, b1, w2 = init_params()                  #Initialize Parameters of the weights and bias
    
    
    for i in range(itera):

        
        net1, a1, net2, a2 = forward_prop(w1, b1, w2, x)                        #forward propagation
        
        dw1, db1, dw2 = back_prop(net1, a1, net2, a2, w1, w2, x, y,m)           #backward propagation
        
        w1, b1, w2 = update_params(w1, b1, w2, dw1, db1, dw2, learning_rate)    #weights and bias updated        
        
        n1, m1 = a2.shape
        new_a2 = np.empty((n1,m1))
        
        for j in range(m1):
            new_a2[:,j] = one_hot_encoding(np.argmax(a2[:,j]))                  #one-hot encoding the output
            
#         print(new_a2)
        if i % 10 == 0:
            print("Iterations : ", i)
            print("Accuracy :", accuracy(new_a2.T, y.T))                        #printing accuracy
            
    return w1, b1, w2

if __name__ == "__main__":

 #importing csv files for train data and labels
    data = pd.read_csv('train_data.csv')
    data_labels = pd.read_csv('train_labels.csv')

    data = np.array(data)
    data_labels = np.array(data_labels)

    m,n = data.shape            

    #splitting the data to 20% test and 80% train
    X_train, X_val, y_train, y_val = train_test_split(data, data_labels, test_size=0.20, random_state=4121)
    #putting the data into arrays
    X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)
    #taking transpose of the matrices
    X_train_tran, X_val_tran, y_train_tran, y_val_tran = X_train.T, X_val.T, y_train.T, y_val.T
    #calling gradient descent function which does a sequence of forward propagation
    #backward propagation and updating weights

    w11, b11, w21 = grad_desc(X_train_tran, y_train_tran, 150, 0.3,m)

    np.save('w11.npy', w11)
    np.save('b11.npy', b11)
    np.save('w21.npy', w21)


    

