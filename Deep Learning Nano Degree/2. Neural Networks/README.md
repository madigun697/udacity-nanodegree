# 2. Neural Networks

In this chapter, instructors teach us the foundations of deep learning and neural networks. 

For basic concepts for deep learning, instructors explain perceptron, loss functions, activation functions, and gradient descent. And instructors also teach useful techniques for training neural networks like early stopping, regularization, dropout, etc.

This chapter includes a mini-project, Sentiment Analysis, with Andrew Trask, the author of Grokking Deep Learning, and the main project, Predicting Bike-Sharing Patterns.

[toc]

**Official course description**

*In this part, you'll learn how to build a simple neural network from  scratch using python. We'll cover the algorithms used to train networks  such as gradient descent and backpropagation.*

*The **first project** is also available this week. In this project, you'll predict bike ridership using a simple neural network.*

*![img](https://video.udacity-data.com/topher/2018/September/5b96d3c7_screen-shot-2018-09-10-at-1.27.33-pm/screen-shot-2018-09-10-at-1.27.33-pm.png)*Multi-layer neural network with some inputs and a single output. Image from [Stanford's cs231n course](http://cs231n.github.io/convolutional-networks/).*

***Sentiment Analysis***

*You'll also learn about model evaluation and validation, an important technique for training and assessing neural networks. We also have  guest instructor Andrew Trask, author of [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning), developing a neural network for processing text and predicting  sentiment. The exercises in each chapter of his book are also available  in his [Github repository](https://github.com/iamtrask/Grokking-Deep-Learning).*

***Deep Learning with PyTorch***

*The last lesson will be all about using the deep learning framework,  PyTorch. Here, you'll learn how to use the Tensor datatype, and the  foundational knowledge you'll need to define and train your own deep  learning models in PyTorch!*

## Foundations of Deep Learning

### Perceptron

As a simplest Feed-Foward Network, perceptron determines the output using input values and their weights. It can be expressed a equation like $$w_1x_1 + w_2x_2 + b = y$$.

<img width="642" alt="image" src="https://user-images.githubusercontent.com/8471958/97819429-53e53a80-1c5d-11eb-9403-733343393d43.png">

### Loss Functions

#### Cross-Entropy

In the classification problem, Cross-Entropy is usually used as a loss function. 

$$CE = -(ylog(p) + 1(1-y)log(1-p))$$ where $$y$$ is actual value(1 or 0), $$p$$ is predicted probability(between 0 and 1).

So, if the actual value is 1, cross-entropy is $$-log(p)$$. And if the actual value is 0, cross-entropy is $$-log(1-p)$$

#### MSE(Mean Squared Error)

In the regression problem, MSE is usually used as a loss function. This is a very simple loss function, the mean of differences between actual and predicted values.

$$MSE = {{1}\over{n}} \Sigma^n_{i=1} {(Y_i - \hat{Y_i})}^2 $$ where $$Y_i$$ is actual value, $$\hat{Y_i}$$ is precited value.

### Activation Functions

#### Sigmoid

The Sigmoid function uses in the binary-classification problem. It returns the value between 0 and 1.

![image](https://user-images.githubusercontent.com/8471958/97820526-75492500-1c63-11eb-91fc-31c6a6c7ab4d.png)

<center><i><p style="font-size:10px">source: http://krisbolton.com/a-quick-introduction-to-artificial-neural-networks-part-2</p></i></center>

#### Softmax

The Softmax function uses in the multi-classification problem. It returns the value between 0 and 1. And the sum of its outputs is 1.

![image](https://user-images.githubusercontent.com/8471958/97819805-b7706780-1c5f-11eb-99c0-fe0f5c144078.png)

<center><i><p style="font-size:10px">source: http://krisbolton.com/a-quick-introduction-to-artificial-neural-networks-part-2</p></i></center>

#### ReLU(Rectified Linear activation function)

One of the purposes of ReLU is to solve the vanishing gradient problem. When using sigmoid as a loss function, the final derivative values too small to update deep layers' weights. However, because ReLU's derivative values are 0 or 1, the derivative values are well backpropagated.

![image](https://user-images.githubusercontent.com/8471958/97820504-5f3b6480-1c63-11eb-94bd-5bd7f5b65aa8.png)

<center><i><p style="font-size:10px">source: http://krisbolton.com/a-quick-introduction-to-artificial-neural-networks-part-2</p></i></center>

### Gradient Descent

To find values(weights), the neural networks modify their weights toward minimization of cost. This algorithm is called Gradient Descent.

![image](https://user-images.githubusercontent.com/8471958/97820364-b1c85100-1c62-11eb-9d78-58db029eca83.png)

<center><i><p style="font-size:10px">source: http://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization/</p></i></center>

## Sentiment Analysis

In this mini-project, I build a simple fully connected neural network(or Feed-Forward Network, FFN). There is one single hidden layer. And this hidden layer size is 100.

![image](https://user-images.githubusercontent.com/8471958/97820653-fef8f280-1c63-11eb-9db1-fc84b85ccb32.png)

Input data is the IMDB's review texts. Before the model train, every review pre-process. Each word in the review texts counts how many appear in the positive reviews and negative reviews in the pre-processing phases. And the ratio of positive to negative is calculated and used as an input data set.

An activation function is sigmoid, and a loss function is a simple error ($$\hat{y} -y$$).

The following image is the result of the training:

![image](https://user-images.githubusercontent.com/8471958/97820892-1e444f80-1c65-11eb-8f7b-5d267cf7700d.png)

## [[Project] Predictin Bike-Sharing Patterns](https://github.com/madigun697/udacity-nanodegree/tree/master/Deep%20Learning%20Nano%20Degree/2.%20Neural%20Networks/Project%201.%20Predicting%20Bike-Sharing%20Patterns)

In this project, you'll get to build a neural network from scratch to carry out a prediction problem on a real dataset! By building a neural  network from the ground up, you'll have a much better understanding of  gradient descent, backpropagation, and other concepts that are important to know before we move to higher-level tools such as PyTorch. You'll  also get to see how to apply these networks to solve real prediction  problems!

The data comes from the [UCI Machine Learning Database](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

