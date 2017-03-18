import tflearn
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences

# Data preparation
int2binary = {}
binary_dim = 8
largest_number = 2**binary_dim
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

a_s = []
b_s = []
c_s = []
for i in range(20000):
	 # get two random binaires a and b and their answer c
	a_int = np.random.randint(largest_number/2) # int version
	b_int = np.random.randint(largest_number/2) # int version
	a = int2binary[a_int] # binary encoding
	b = int2binary[b_int] # binary encoding
	c = int2binary[a_int+b_int] # binary encoding - answer

	a_s += list(a)
	b_s += list(b)
	c_s += list(c)

trainX = np.array([a_s, b_s]).T
trainY = np.array([c_s]).T



# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
print(trainX,trainY, ...)

# Network building
net = tflearn.input_data([None, 2])
net = tflearn.embedding(net,input_dim=10000, output_dim=128)
net = tflearn.simple_rnn(net, 16)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=None, show_metric=True,
          batch_size=32)
