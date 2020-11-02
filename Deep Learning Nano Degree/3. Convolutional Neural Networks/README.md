# 3. Convolutional Neural Networks

This chapter teaches us one of the famous deep learning architecture, **Convolutional Neural Networks(CNNs)**.

This chapter includes not only how to construct CNNs using PyTorch but also transfer learning and autoencoders.

[toc]

**Official course description**

*Convolutional networks have achieved state of the art results in computer vision.  These types of networks can detect and identify objects in images.  You'll learn how to build convolutional networks in PyTorch.*

*You'll also get the **second project,** where you'll build a convolutional network to classify dog breeds in pictures.*

*![img](https://video.udacity-data.com/topher/2018/September/5b96d517_screen-shot-2018-09-10-at-1.33.11-pm/screen-shot-2018-09-10-at-1.33.11-pm.png)*

*Structure of a convolutional neural network.*

*You'll also use convolutional networks to build an autoencoder, a network architecture used for image compression and denoising. Then,  you'll use a pre-trained neural network, to classify images the network  has never seen before, a technique known as transfer learning.*

## Convolutional Neural Networks(CNNs)

### Basic structure of CNN

![image](https://user-images.githubusercontent.com/8471958/97821530-9dd31e00-1c67-11eb-843e-9dd78494acb1.png)

#### Convolutional layer

The convolutional layer projects the input on output using a filter. 

![image](https://user-images.githubusercontent.com/8471958/97821599-e4c11380-1c67-11eb-8a9b-7334021eddff.png)

<center><i><p style="font-size:10px">source: https://www.researchgate.net/figure/Outline-of-the-convolutional-layer_fig1_323792694</p></i></center>

![image](https://user-images.githubusercontent.com/8471958/97821695-349fda80-1c68-11eb-9e24-f7711a6a49f3.png)

<center><i><p style="font-size:10px">source: https://www.quora.com/What-are-convolutional-layers</p></i></center>

#### Pooling layer

To avoid over-fitting and reduce features, extracts max(or in) value in the feature map.

![image](https://user-images.githubusercontent.com/8471958/97821871-c90a3d00-1c68-11eb-93c0-bf2ac7363293.png)

<center><i><p style="font-size:10px">source: https://kevinthegrey.tistory.com/142</p></i></center>

### Code Samples

#### Define Network

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        # output shape is (20, 16, 32, 32)
        # 32+(2x2) - (5-1) = 32 + 4 - 4 = 32
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        # max pooling layer
        # output shape is (20, 16, 16, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        # convolutional layer
        # output shape is (20, 32, 16, 16)
        # 16+(2x2) - (5-1) = 16 + 4 - 4 = 16
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        # max pooling layer
        # output shape is (20, 32, 8, 8)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64*4*4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

#### Training 

```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        start = time.time()        
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

    # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        print('Time consumed %.4f seconds' % (time.time() - start))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Valid loss decreased (%.6f --> %.6f).' % (valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
  
model = train(50, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, use_cuda, 'model_scratch.pt')
```

#### Testing

```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
```

## Transfer Learning

Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set. 

Depending on both:

- The size of the new data set, and
- The similarity of the new data set to the original data set

The approach for using transfer learning will be different. There are four main cases:

1. New data set is small, new data is similar to original training data.
2. New data set is small, new data is different from original training data.
3. New data set is large, new data is similar to original training data.
4. New data set is large, new data is different from original training data.

A large data set might have one million images. A small data could have  two-thousand images. The dividing line between a large data set and  small data set is somewhat subjective. Overfitting is a concern when  using transfer learning with a small data set. 

Images of dogs and images of wolves would be considered similar; the  images would share common characteristics. A data set of flower images  would be different from a data set of dog images. 

Each of the four transfer learning cases has its own approach. In the following sections, we will look at each case one by one.

The graph below displays what approach is recommended for each of the four main cases. 

![img](https://video.udacity-data.com/topher/2018/September/5baa60db_screen-shot-2018-09-25-at-9.22.35-am/screen-shot-2018-09-25-at-9.22.35-am.png)

## Autoencoder

Convolutional Autoencoder consists of the encoder part and the decoder part. The encoder part is a typical convolutional network. This part compresses information of original input. The decoder part is a reverse of the typical convolutional network. The decoder uses Transpose Convolution layers to convert from compressed vector to original information.

Style transfer project is a good example using autoencoder. Decoder trained by other style images can convert compressed information using the encoder. In this case, the original image converts the same image but different styles.

![image](https://user-images.githubusercontent.com/8471958/97822460-e8a26500-1c6a-11eb-8c9b-a2bc78e3737a.png)

<center><i><p style="font-size:10px">source: https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694</p></i></center>

## [[Project] Dog-Breed Classifier](https://github.com/madigun697/udacity-nanodegree/tree/master/Deep%20Learning%20Nano%20Degree/3.%20Convolutional%20Neural%20Networks/Project%202.%20Dog%20Breed%20Classifier)

Welcome to the Convolutional Neural Networks (CNN) project! In this  project, you will learn how to build a pipeline to process real-world,  user-supplied images.  Given an image of a dog, your algorithm will  identify an estimate of the canine’s breed.  If supplied an image of a  human face, the code will identify the resembling dog breed.  

Along with exploring state-of-the-art CNN models for classification,  you will make important design decisions about the user experience for  your app.  By completing this lab, you demonstrate your understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  

> Each model has its strengths and weaknesses, and engineering a  real-world application often involves solving many problems without a  perfect answer.
> Your imperfect solution will nonetheless create a fun user experience!