import numpy as np
import pandas as pd

'''
Using the MNIST dataset we will write a neural network with 1 hidden layer to predict the handwritten digits. 
We will only use NumPy.
'''

# Load dataset
train_data = pd.read_csv('/Users/akashdas/Downloads/archive 2/mnist_train.csv')
test_data = pd.read_csv('/Users/akashdas/Downloads/archive 2/mnist_test.csv')

train_data = np.array(train_data) # 60000, 785
test_data = np.array(test_data)

np.random.shuffle(train_data)

x_train = train_data[:50000, 1:] / 255.0 # 50000, 784
y_train = train_data[:50000, 0] # 50000, 1

x_dev = train_data[50000:, 1:] / 255.0 # 10000, 784
y_dev = train_data[50000:, 0] # 10000, 785

x_test = test_data[:, 1:] / 255.0 # 10000, 784
y_test = test_data[:, 0] # 10000, 1

# Ititializations
def init_params():

    '''
    Initialize paramters to the network. 
    Create small initializations to prevent vanishing or exploding gradients
    '''
    
    weight1 = np.random.randn(784, 128) * 0.01  # 784 input features to 128 hidden units
    bias1 = np.zeros((1, 128))                  # 128 hidden units
    weight2 = np.random.randn(128, 60) * 0.01   # 128 hidden units to 60 output units
    bias2 = np.zeros((1, 60))                   # 60 output units
    weight3 = np.random.randn(60, 10) * 0.01    # 60 hidden units to 10 output units
    bias3 = np.zeros((1, 10))                   # 10 output units

    return weight1, bias1, weight2, bias2, weight3, bias3

# ReLU
def ReLU(z):
    return np.maximum(0, z)

def ReLU_deriv(z):
    return np.where(z > 0, 1, 0)

# Softmax
def softmax_activation(z):
    z = z - np.max(z, axis=1, keepdims=True)  # Numerical stability
    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward propagation
def fwd_prop(x_train, weight1, bias1, weight2, bias2, weight3, bias3):
    z1 = np.dot(x_train, weight1) + bias1
    #print(f"z1 shape: {z1.shape}")  # Should be (50000, 128)
    fz1 = ReLU(z1)
    #print(f"fz1 shape: {fz1.shape}")  # Should be (50000, 128)
    z2 = np.dot(fz1, weight2) + bias2
    #print(f"z2 shape: {z2.shape}")  # Should be (50000, 60)
    fz2 = ReLU(z2)
    #print(f"fz2 shape: {fz2.shape}")  # Should be (50000, 60)
    z3 = np.dot(fz2, weight3) + bias3
    #print(f"z3 shape: {z3.shape}")  # Should be (50000, 10)
    fz3 = softmax_activation(z3)
    #print(f"fz3 shape: {fz3.shape}")  # Should be (50000, 10)

    return z1, fz1, z2, fz2, z3, fz3

# One-hot encode true labels
def one_hot_encoding(y_train, num_classes=10): 

    '''
    Create a zero-padded matrix unless the positional index contains a number
    y-train - y labels from training data
    num_classes - arg to denote number of possible outputs from data (0-9)
    '''

    encoded_arr = np.zeros((y_train.size, num_classes), dtype=int)
    encoded_arr[np.arange(y_train.size), y_train] = 1

    return encoded_arr

# Cross-entropy loss
def cross_entropy(fz3, y_train):

    '''
    Calculate loss between output layer and true labels.
    fz3 - output layer
    y_train - true labels
    '''

    epsilon = 1e-10  # Small constant to avoid log(0)
    m = y_train.shape[0]  # Normalization constant

    return -np.sum(y_train * np.log(fz3 + epsilon)) / m

# Backpropagation
def back_prop(x_train, z1, weight1, bias1, fz1, z2, weight2, bias2, fz2, z3, weight3, bias3, fz3, y_train):
    m = y_train.shape[0]
    
    # Gradient of the loss with respect to fz3
    dz3 = fz3 - y_train

    # Gradient of the loss with respect to weight3 and bias3
    dw3 = (1/m) * np.dot(fz2.T, dz3)
    db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)

    # Gradient of the loss with respect to fz2
    dz2 = np.dot(dz3, weight3.T) * ReLU_deriv(z2)

    # Gradient of the loss with respect to weight2 and bias2
    dw2 = (1/m) * np.dot(fz1.T, dz2)
    db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

    # Gradient of the loss with respect to fz1
    dz1 = np.dot(dz2, weight2.T) * ReLU_deriv(z1)

    # Gradient of the loss with respect to weight1 and bias1
    dw1 = (1/m) * np.dot(x_train.T, dz1)
    db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3

def update_params(weight1, bias1, weight2, bias2, weight3, bias3, dw1, db1, dw2, db2, dw3, db3, alpha):
    weight1 -= alpha * dw1
    bias1 -= alpha * db1
    weight2 -= alpha * dw2
    bias2 -= alpha * db2
    weight3 -= alpha * dw3
    bias3 -= alpha * db3

    return weight1, bias1, weight2, bias2, weight3, bias3

def accuracy(fz3, y_label):
    predicted_classes = np.argmax(fz3, axis=1)
    true_classes = np.argmax(y_label, axis=1)

    return np.mean(predicted_classes == true_classes)

# Learning rate testing
alphas = np.arange(1e-15, 1, 0.005)
accuracy_arr = np.zeros((alphas.shape))

def train_model(x_train, y_train, weight1, bias1, weight2, bias2, weight3, bias3, alpha, epochs):

    '''
    Train model via Gradient Descent
    '''

    y_train_encoded = one_hot_encoding(y_train)
    
    for epoch in range(epochs):
        z1, fz1, z2, fz2, z3, fz3 = fwd_prop(x_train, weight1, bias1, weight2, bias2, weight3, bias3)
        losses = cross_entropy(fz3, y_train_encoded)
        acc = accuracy(fz3, y_train_encoded)
        dw1, db1, dw2, db2, dw3, db3 = back_prop(x_train, z1, weight1, bias1, fz1, z2, weight2, bias2, fz2, z3, weight3, bias3, fz3, y_train_encoded)
        weight1, bias1, weight2, bias2, weight3, bias3 = update_params(weight1, bias1, weight2, bias2, weight3, bias3, dw1, db1, dw2, db2, dw3, db3, alpha)

        if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {losses}")
                print(f"Epoch {epoch} | Accuracy: {acc}")
    
    accuracy_arr[-1] = acc
    #print(f"Epoch {epoch} | Loss: {losses}")
    print(f"Epoch {epoch} | Accuracy: {acc}")

    return weight1, bias1, weight2, bias2, weight3, bias3
    
# --- Run Model ---

# Initialize parameters
weight1, bias1, weight2, bias2, weight3, bias3 = init_params()

# Train model
weight1, bias1, weight2, bias2, weight3, bias3 = train_model(x_train, y_train, weight1, bias1, weight2, bias2, weight3, bias3, alpha = 0.055000000000001, epochs = 4000)

# Evaluate on test data
_, _, _, _, _, fz3_test = fwd_prop(x_test, weight1, bias1, weight2, bias2, weight3, bias3)
y_test_encoded = one_hot_encoding(y_test)  # One-hot encode y_test for evaluation
print(f"Test Accuracy: {accuracy(fz3_test, y_test_encoded) * 100}%")

"""
def train_model_alpha_test(x_train, y_train, weight1, bias1, weight2, bias2, alphas, epochs):

    '''
    Train model via Gradient Descent
    '''

    y_train_encoded = one_hot_encoding(y_train)
    
    for epoch in range(epochs):
        z1, fz1, z2, fz2, z3, fz3 = fwd_prop(x_train, weight1, bias1, weight2, bias2, weight3, bias3)
        losses = cross_entropy(fz3, y_train_encoded)
        acc = accuracy(fz3, y_train_encoded)
        dw1, db1, dw2, db2, dw3, db3 = back_prop(x_train, z1, weight1, bias1, fz1, z2, weight2, bias2, fz2, z3, weight3, bias3, fz3, y_train_encoded)
        weight1, bias1, weight2, bias2, weight3, bias3 = update_params(weight1, bias1, weight2, bias2, weight3, bias3, dw1, db1, dw2, db2, dw3, db3, alpha)

        if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {losses}")
                print(f"Epoch {epoch} | Accuracy: {acc}")

    accuracy_arr[-1] = acc
    print(f"Epoch {epoch} | Accuracy: {acc}")
    
    return weight1, bias1, weight2, bias2
"""