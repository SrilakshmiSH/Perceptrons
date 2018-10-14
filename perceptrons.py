
'''
Programming Assignment-1
Two-layer neural network - Neural network with 784 inputs, 1 hidden layer having n hidden units(20,50,100)
and 10 output units. 
The hidden layer uses sigmoid activation function. 
Every input connects to one hidden unit and every hidden unit connects to one output.
'''

import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import *

learningrate = 0.1

# vary momentum for experiment-2
momentum1 = 0.9
momentum2 = 0
momentum3 = 0.25
momentum4 = 0.5

# vary hidden units for experiment-1
hiddenunits1 = 20
hiddenunits2 = 50
hiddenunits3 = 100

# read train dataset from csv file
input_data1 = csv.reader(open("/Users/srilakshmishivakumar/cs554 ml/hw1/mnist_train.csv"))
a = list(input_data1)
traindata = np.array(a).astype("float")

# preprocessing
traindata[:,1:]/= 255

trlabels = traindata[:, :1]
trainlabels = np.full((60000,10),0.1,dtype=float)

traindata = traindata[:, 1:]
trainones = np.ones((60000,1))
train = np.append(trainones,traindata,axis=1)

for i in range(len(trlabels)):
	idi = trlabels[i].astype(int)
	trainlabels[i][idi[0]] += 0.8

# read test dataset from csv file
input_data2 = csv.reader(open("/Users/srilakshmishivakumar/cs554 ml/hw1/mnist_test.csv"))
b = list(input_data2)
testdata = np.array(b).astype("float")

# preprocessing
testdata[:,1:]/= 255
telabels = testdata[:, :1]
testlabels = np.full((10000,10),0.1,dtype=float)

testdata = testdata[:, 1:]
testones = np.ones((10000,1))
test = np.append(testones,testdata,axis=1)

for i in range(len(telabels)):
	idi = telabels[i].astype(int)
	testlabels[i][idi[0]] += 0.8


# weights for input layer and hidden layer
w_hiddenlayer = np.random.uniform(-0.05, 0.05, (785,hiddenunits3))
w_hiddenlayer_withbias = np.random.uniform(-0.05,0.05,(hiddenunits3+1,10))

# weights from hidden layer to output unit
w_hd_to_output = np.zeros((hiddenunits3+1,10))
# weights from input unit to hidden layer 
w_input_to_hd = np.zeros((hiddenunits3,785))

# activation of hidden node j
hj = np.zeros(hiddenunits3+1)
hj[0] = 1

# predicted train and test accuracies
predicted_train_accuracy = np.zeros((60000,1))
predicted_test_accuracy = np.zeros((10000,1))

# final train and test accuracies
train_accuracy = np.zeros(50)
test_accuracy = np.zeros(50)

# steps for backpropagation

for i in range(50):
	for j in range(len(train)):
# step 1 - to calculate activation hj and activation ok	
		wjixi = np.dot(train[j][:],w_hiddenlayer)
		hj[1:] = 1 / (1 + np.exp(-wjixi))
		wkjhj = np.dot(hj,w_hiddenlayer_withbias)
		ok = 1 / ( 1 + np.exp(-wkjhj))

# step 2 - error term for output unit k and error term for hidden unit j
		output_error = ok * (1 - ok) * (trainlabels[j][:] - ok)
		a = hj[1:] * (1 - hj[1:])
		b = np.dot(w_hiddenlayer_withbias[1:,:],np.transpose(output_error))
		hiddenunit_error = a * b

#step 3 - update weights with momentum 
		x = learningrate * np.outer(hj,output_error)
		y = momentum1 * w_hd_to_output
	# update weights from hidden to output layer 	
		new_hd_to_output = x + y

		x = learningrate * np.outer(hiddenunit_error,train[j][:])
		y = momentum1 * w_input_to_hd
	# output weights from input to hidden layer	
		new_input_to_hd = x + y

	# keep weight changes moving in sme direction 	
		w_hiddenlayer_withbias += new_hd_to_output
		w_hd_to_output = new_hd_to_output
		w_hiddenlayer += np.transpose(new_input_to_hd)
		w_input_to_hd = new_input_to_hd

		idi = np.argmax(ok, axis=0)
		predicted_train_accuracy[j] =  idi

	# calculate confusion matrix for train dataset	
	train_cm = confusion_matrix(trlabels,predicted_train_accuracy)
	#obtain diagonal sum
	train_diagonalsum =  sum(np.diag(train_cm))
	# plot the graph using the diagonal sum accuracy
	train_accuracy[i] = (train_diagonalsum/60000.00) * 100
	

	for j in range(len(test)):
	# calculate the activation hj and  activation ok for test dataset	
		wjixi = np.dot(test[j][:],w_hiddenlayer)
		hj[1:] = 1 / (1 +  np.exp(-wjixi))
		wkjhj = np.dot(hj,w_hiddenlayer_withbias)
		ok = 1 / (1 + np.exp(-wkjhj))
		
		idi = np.argmax(ok, axis=0)
		predicted_test_accuracy[j] =  idi

	# calculate confusion matrix for test dataset
	test_cm = confusion_matrix(telabels,predicted_test_accuracy)
	#obtain diagonal sum
	test_diagonalsum = sum(np.diag(test_cm))
	# plot the graph using accuracy of the diagonal sum
	test_accuracy[i] = (test_diagonalsum/10000.00) * 100


print("\n momentum - 0.9")
print("\nConfusion matrix for train data with 100 hidden data points\n")
print(train_cm)
print("\nTrain data accuracy\n")
print(train_accuracy)
print("\nConfusion matrix for test data with 100 hidden data points\n")
print(test_cm)
print("\nTest data accuracy\n")
print(test_accuracy)

plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

figure = "hd100.png"
plt.title("Accuracy for Neural Network with 100 hidden data units")
plt.savefig(figure)
plt.show()




