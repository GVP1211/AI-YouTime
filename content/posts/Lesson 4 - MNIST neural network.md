+++
title = 'Lesson 4 - MNIST neural network'
date = '2024-10-27'
+++

# Making your first neural network from scratch

For the first model, we will be making a model that can recognise hand written numbers using the MNIST dataset.

## Designing the model
Firstly, the MNIST database contains 70000 handwritten numbers in a grey-scale 28 x 28 bitmap image. The database will have a 2-D array containing the pixels each with a value between 0 to 255 on the grey-scale and a seperate array of data with the labels.
This means there are 784 pixels in total per input data, therefore we will have 784 input nodes.

Next, we will have 1 hidden layer for simplicity's sake, this layer will have 40 nodes. Then we will have an output layer of 10 nodes. Each output node corresponding to a digit from 0 to 10.

### Initialisation

1. Firstly, we will fetch the data from the MNIST data base, we will assign the bitmap images to X and X_test, the labels as Y and Y_test.
2. Initialising weights and biases; the weights between input and layer 2 will be names W1 and the size [40, 784], W2 for layer 2 and output layer will have a size of [10, 40]. We will also have a bias for the input layer and layer 2 named B1 and B2 with size [1, 40] and [1, 10] respectively.

### Forward propagation
1. When propagating from the input layer to the second layer we will take the dot product of X and W1 then add it to B1 to obtain Z1. The dot product takes every single bitmap image and multiplies each of them by our weight.
2. Then we need to put Z1 through the activation function. For this model, we will use a sigmoid curve which is the function f(x) = 1/(1-e^-x). The result is A1 which is our second layer.
3. We then take the dot product of A1 and W2 then add to B2 to obtain our output layer which is called Z2. We can then put Z2 through an activation function called softmax function. A softmax function takes a set of numbers and changes them to a probability based on how high each of the numbers were.

### Backward propagation
This section can be skipped if you don't understand calculus and the chain rule.
1. Firstly, we calculate the differentiated value of Z2 or dZ2 which is dZ2 = A2 - one_hot(Y). one_hot(Y) means we are creating an array where all the values are 0 and the index of Y is 1. However, we need to divide dZ2 by the size of Y as we will be summing dZ2 for our bias.
2. Next we differentiate W2 in terms of Z2 so we get A1, then applying the chain rule to get dW2 in terms of the loss, we multiply with dZ2. So we get dW2 = dZ2 x A1
3. When diffrentiating B2 in terms of Z2 we get 1, so applying the chain rule again to get dB2 in terms of the loss, we get dB2 = sum of Z2.
4. To differentiate A1 in terms of Z2 we get W2, applying the chain rule to get dA1 in terms of the loss, we multiple W2 with dZ2. So we get dA1 = W2 x dZ2.
5. Next we need to find the derivative of Z1, since A1 = sigmoid(Z1), we need to find the derivative of the sigmoid function which is f(x) x (1-f(x)) where f(x) is the sigmoid function. Let deriv_sigmoid(x) be the derivative of f(x). Then to find dZ1 we use the chain rule and get dZ1 = dA1 x deriv_sigmoid(Z1).
6. Similar to step 2 and 3, we get dW1 = X * dZ1 and dB1 = sum of dZ1.

### Gradient descent
In gradient descent there are 4 main steps:
1. Forward propagation.
2. Backward propagation
3. Update parameters (weights and biases). For this we take our original weights and biases then subtract by the respective derivatives multiplied by the learning rate.
4. Track our accuracy on both train and test data.
These 4 steps will looped a certain amount of time until the model is accurate enough to be used.

## Coding
This code will be done in google colab.

### Initialisation
``` python
path = '/tmp'
def fetch(url): #This code can be ignored, it is just fetching data from a online file
  fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      data = f.read()
  else:
    with open(fp, "wb") as f:
      data = requests.get(url).content
      f.write(data)
  return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("https://github.com/sunsided/mnist/raw/master/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).T #shaping data into a 2d array of size [x, 784], x is whatever number to make the data fit (60000). Then .T is to transpose the data from rows to columns and vice versa. This makes the data easier to work with.
Y = fetch("https://github.com/sunsided/mnist/raw/master/train-labels-idx1-ubyte.gz")[8:].T.astype('int') #fetching labels, transpose and change to int data type
X_test = fetch("https://github.com/sunsided/mnist/raw/master/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).T
Y_test = fetch("https://github.com/sunsided/mnist/raw/master/t10k-labels-idx1-ubyte.gz")[8:].T.astype('int')
```

``` python
X = X / 255.0 # since data is between 0 and 255, we divide it by 255 to have the values between 0 and 1, this is called data normalisation.
X_test = X_test / 255.0
```

``` python
def init_params(): # Initialiing all weights and parameters as a random number between -0.5 and 0.5 (data type double is just a float number that can have more significant figures)
  W1 = (np.random.rand(40, 784) - 0.5).astype('double')
  b1 = (np.random.rand(40, 1) - 0.5).astype('double')
  W2 = (np.random.rand(10, 40) - 0.5).astype('double')
  b2 = (np.random.rand(10, 1) - 0.5).astype('double')
  return W1, b1, W2, b2
```

### Forward propagation
``` python
def sigmoid(Z): #Activation function is to increase complexity and make the model non-linear, it also compresses all values to between 0 and 1
  return 1/(1+np.exp(-Z))

def softmax(Z): #Softmax function reduces all numbers in an array to a probability between 0 and 1, sum of all elements should be 1
  exp_element=np.exp(Z-Z.max())
  return exp_element/np.sum(exp_element,axis=0)

def forward_prop(W1, b1, W2, b2, X): #forward propagation
  Z1 = W1.dot(X) + b1
  A1 = sigmoid(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2
```

### Backward propagation
``` python
def deriv_sigmoid(Z):
  return sigmoid(Z)*(1-sigmoid(Z)) #Derivative of the sigmoid function

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y): #Back propagation, calculus
  dZ2 = (A2 - one_hot(Y))/Y.size
  dW2 = dZ2.dot(A1.T)
  db2 = np.sum(dZ2, axis = 1, keepdims = True)
  dA1 = W2.T.dot(dZ2)
  dZ1 = dA1 * deriv_sigmoid(Z1)
  dW1 = dZ1.dot(X.T)
  db1 = np.sum(dZ1, axis = 1, keepdims = True)
  return dW1, db1, dW2, db2 #These values tell how much each weight and bias has to be adjusted by

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr): #Update parameters, lr is the learning rate
  W1 = W1 - lr * dW1
  b1 = b1 - lr * db1
  W2 = W2 - lr * dW2
  b2 = b2 - lr * db2
  return W1, b1, W2, b2
```

### Gradient descent
``` python
def get_predictions(A2): #Gives an output prediction
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y): #Calculates accuracy based on all predictions and all lables
  return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, X_test, Y_test, lr, epochs, accuracies, test_accuracies, W1 = [], b1 = [], W2 = [], b2 = []): #Training the model
  for i in range(epochs):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) #forward propagation
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y) #back propagation
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr) #update weights and biases
    predictions = get_predictions(A2)
    accuracies.append(get_accuracy(predictions, Y)) #Keep track of accuracy whilst training
    if i % 50 == 0:
      print("Iteration: ", i)
      print(get_accuracy(predictions, Y)) #prints accuracy whilst training every 50 loops
  return W1, b1, W2, b2, A2, accuracies, test_accuracies

accuracies = []
test_accuracies = []
W1, b1, W2, b2, A2, accuracies, test_accuracies = gradient_descent(X, Y, X_test, Y_test, 0.70, 200, accuracies, test_accuracies)
```

### Testing our model
``` python
def make_predictions(X, W1, b1, W2, b2): #Get prediction based on a dataset
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2): #Get prediction based on 1 image from test dataset as well as plot image
    current_image = X_test[:, index, None]
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
    label = Y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

predictions = make_predictions(X_test, W1, b1, W2, b2) #Get accuracy from the test/ validation data
test_accuracy = np.sum(predictions == Y_test) * 100 / Y_test.size
print("Accuracy = ", test_accuracy ,"%")
```
``` python
test_prediction(93, W1, b1, W2, b2) # Test from test image dataset

from skimage.draw import line_aa 
img = np.zeros((14, 14), dtype=np.uint8)
rr, cc, val = line_aa(4, 4, 2, 6)
img[rr, cc] = val * 255
rr, cc, val = line_aa(2, 6, 4, 8)
img[rr, cc] = val * 255
rr, cc, val = line_aa(4, 8, 10, 4)
img[rr, cc] = val * 255
rr, cc, val = line_aa(10, 4, 10, 8)
img[rr, cc] = val * 255
n_test = np.zeros((28, 28))
for i in range(13):
  for j in range(13):
    n_test[2*i][2*j] = img[i][j]
    n_test[2*i+1][2*j] = img[i][j]
    n_test[2*i][2*j+1] = img[i][j]
    n_test[2*i+1][2*j+1] = img[i][j]
n_test = n_test/255
n_test = n_test.reshape(-1, 1)
imshow(n_test.reshape(28,28))
prediction2 = make_predictions(n_test, W1, b1, W2, b2)
print(prediction2)

n = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 2, 8, 7, 3, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 2, 7, 2, 7, 8, 5, 0, 0, 0, 0, 0],
     [0, 0, 0, 9, 4, 0, 0, 2, 8, 2, 0, 0, 0, 0],
     [0, 0, 3, 6, 0, 0, 0, 0, 9, 2, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 8, 2, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 3, 8, 2, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 7, 7, 2, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 2, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 7, 8, 9, 7, 7, 7, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
n_test = np.zeros((28, 28))
for i in range(13):
  for j in range(13):
    n_test[2*i][2*j] = n[i][j]
    n_test[2*i+1][2*j] = n[i][j]
    n_test[2*i][2*j+1] = n[i][j]
    n_test[2*i+1][2*j+1] = n[i][j]
n_test = n_test/10
n_test = n_test.reshape(-1, 1)
imshow(n_test.reshape(28,28))
prediction2 = make_predictions(n_test, W1, b1, W2, b2)
print(prediction2)
```

### Exercises
Once the code is completed, here are some challenges you could do:
1. Keep track of the test accuracy as the model is being trained.
2. Try different things to increase accuracy as high as possible
3. Are there any other different activation or loss functions you could use?

### References
- https://www.youtube.com/watch?v=w8yWXqWQYmU&t=532s
- https://www.geeksforgeeks.org/ml-one-hot-encoding/
- https://davidbieber.com/snippets/2020-12-12-derivative-of-softmax-and-the-softmax-cross-entropy-loss
- https://github.com/sunsided/mnist

### Further reading
- https://www.ibm.com/topics/overfitting