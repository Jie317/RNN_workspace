import copy
import numpy as np
np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_output_to_derivatie(output):
    return output*(1-output)

# training dataset generation
int2binary = {}
binary_dim = 8
largest_number = 2**binary_dim
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# hyperparams
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# weights
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1 # 2*16
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1 # 16*16
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1 # 16*1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training 10000 samples
for j in range(20000):
    # get two random binaires a and b and their answer c
    a_int = np.random.randint(largest_number/2) # int version
    b_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding
    b = int2binary[b_int] # binary encoding
    c = int2binary[a_int+b_int]

    d = np.zeros_like(c)

    overall_error = 0
    layer_2_deltas = []
    layer_1_values = [] # hidden state
    layer_1_values.append(np.zeros(hidden_dim))

    for pos in range(binary_dim):
        X = np.array([ [a[binary_dim-pos-1], b[binary_dim-pos-1]] ]) # 1*2
        y = np.array([[c[binary_dim-pos-1]]]) # 1*1

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], 
            synapse_h)) # 1*16 hidden vector

        layer_2 = sigmoid(np.dot(layer_1, synapse_1)) # 1*1 output
        layer_2_error = y - layer_2

        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivatie(
            layer_2))

        overall_error += np.abs(layer_2_error[0]) # total errors

        d[binary_dim-pos-1] = np.round(layer_2[0][0])

        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    # backpropagation through time
    for pos in range(binary_dim):
        X = np.array([[a[pos], b[pos]]])
        layer_1 = layer_1_values[-pos-1]
        prev_layer_1 = layer_1_values[-pos-2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-pos-1]

        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + 
            layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivatie(
            layer_1)

        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # update
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
    if j%1000 == 0:
        print('Iteration: ', j)
        print('Error: ', overall_error)
        print('Pred: ', d)
        print('True: ', c)
        #print(synapse_0, synapse_h, synapse_1)
        out = 0
        for i,x in enumerate(reversed(d)):
            out += x*2**i
        print(a_int, '+', b_int, '=', out, tuple(d)==tuple(c))
        print('----------\n')








