

# Table of contents
- [Table of contents](#table-of-contents)
- [Overview of the project](#overview-of-the-project)
  - [Architecture](#architecture)
  - [Calculating Biases and Weights](#calculating-biases-and-weights)
  - [Prerequisites](#prerequisites)
    - [Modules](#modules)
    - [The Dataset](#the-dataset)
  - [Things I'm still working on](#things-im-still-working-on)


# Overview of the project    
This is my investigation into Neural Networks and the math behind them. This neural network is designed to learn the MNIST dataset and be able to recognize handwritten digits. 
The purpose of this project is to create a neural network without using Keras or Tensorflow and without using dataset manipulation modules like Pandas. In this way, I can try to gain a better knowledge of how neural networks actually work without using an abstracted version similar to what is used in Tensorflow



## Architecture 
- The network will have 784 input nodes (one for every pixel in the 28x28 images)
- The network will have 10 output nodes (representing every possible output. Ie: 0,1,2,3,4,5,6,7,8,9)
- The amount of hidden layer nodes is up to you! However it's important to know that the more complex the network is, the higher the possibility for a high accuracy. So keeping this in mind, I might recommend a few hidden layers with a few dozen nodes in each. 


## Calculating Bias and Weight Derivatives
In order to calculate the bias and weight derivatives, you must first calculate the BASE derivative which is used in the calculation of both the biases and the weights

To see how I calculated the base derivatives you can view these photos:
[Page 1](https://1drv.ms/u/s!AuuhftLL-JDsgpYkdrXvJQGa59dcRg?e=huE1Y8)
[Page 2](https://1drv.ms/u/s!AuuhftLL-JDsgpYj4t_DphXmlk10Hg?e=o7uOyv)

## Prerequisites 
### Modules
There is one module used in this program that does not come as default in python:\
\
**Numpy:**
```
pip install numpy
```
### The Dataset
The csv files containing the training and testing data can be installed through these links:

[Training](https://python-course.eu/data/mnist/mnist_train.csv) (60k training examples)
[Testing](https://python-course.eu/data/mnist/mnist_test.csv) (10k testing examples)

**Ensure that the dataset files are in the same directory as the Neural Network program**
## Things I'm still working on
There are many variables that affect the network (Amount of epochs, batch size, learning rate, amount of hidden layers, amount of nodes in the hidden layers)
I'm currently testing how I can improve the accuracy by tweaking these variables. When I find better values for the variables listed, I will update the code above to reflect the highest accuracy I have been able to achieve.  
(Highest accuracy obtained has been 93%)

