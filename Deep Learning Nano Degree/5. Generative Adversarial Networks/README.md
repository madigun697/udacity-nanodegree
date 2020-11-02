# 5. Generative Adversarial Networks

Generative Adversarial Networks(GANs) consist of a Generator and a Discriminator. It seems like cops and counterfeiters.

These two parts train themselves in the opposite direction. The generator(Counterfeiters) trains to create fake input. While the discriminator(Cops) trains to detect and filter out fake inputs. The purposes of using GANs vary. Most of them use the generator to create new images, sentences, even music.

[toc]

**Official course description**

*Generative adversarial networks (GANs) are one of the newest and most exciting  deep learning architectures, showing incredible capacity for  understanding real-world data. In this part, you'll learn about and  implement GANs for a variety of tasks. You'll even see how to code a [CycleGAN](https://github.com/junyanz/CycleGAN) for generating images, and learn from one of the creators of this formulation, Jun-Yan Zhu, a researcher at [MIT's CSAIL](https://www.csail.mit.edu/).*

![img](https://video.udacity-data.com/topher/2018/September/5b96e7c1_screen-shot-2018-09-10-at-2.52.51-pm/screen-shot-2018-09-10-at-2.52.51-pm.png)

*Examples of image-to-image translation done by [CycleGAN and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) formulations.*

*The inventor of GANs, Ian Goodfellow, will show you how GANs work and how to implement them. Then, in the **fourth project,** you'll use a deep convolutional GAN to generate completely new images of human faces.*

*![img](https://video.udacity-data.com/topher/2018/September/5b96e835_screen-shot-2018-09-10-at-2.54.49-pm/screen-shot-2018-09-10-at-2.54.49-pm.png)*

*Low-res, GAN-generated images of faces.*

## Generative Adversarial Networks(GANs)

### Structure of GANs

![image](https://user-images.githubusercontent.com/8471958/97830603-5e65fb00-1c82-11eb-8dac-e1f6778f1677.png)

#### Generator

The generator looks like the decoder in Autoencoder. It creates the new sample data from the latent sample vector $$z$$. $$z$$ is the specific size vector with random values. The generator network is not limited. It can use the FFNs(Feed-Forward Networks) or Transpose Convolutional layers like decoder.

#### Discriminator

The Discriminator is a binary classification network. This network's purpose is to determine whether input data is real or not(fake, generated).

#### Loss Function

The discriminator losses are the sum of two losses. The one is real losses when input data is real data; another is fake losses when input data is generated. These two losses both use BCEWithLogitsLoss that is a combined sigmoid activation function and binary cross-entropy loss.

The generator losses and the real losses for discriminator are the same. However, the generator losses use the flipped labels. It means that input is a generated image, and the label is real.

To sum up, the discriminator losses mean that how well does discriminator choose the right things. In contrast, the generator losses mean how well the generator creates new data to trick the discriminator. 

### Code Samples

#### Define Network

##### Discriminator

```python
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        # flatten image
        x = x.view(-1, 28*28)
        
        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = self.dropout(x)
        x = self.fc4(x)

        return x
```

##### Generator

```python
class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        
        self.dropout = nn.Dropout(0.3)        

    def forward(self, x):
        # pass x through all layers
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = self.dropout(x)
        x = self.fc4(x)        
        # final layer should have tanh applied
        x = F.tanh(x)
        
        return x
```

##### Building Network

```python
# Discriminator hyperparams

# Size of input image to discriminator (28*28)
input_size = 784
# Size of discriminator output (real or fake)
d_output_size = 1
# Size of *last* hidden layer in the discriminator
d_hidden_size = 64

# Generator hyperparams

# Size of latent vector to give to generator
z_size = 100
# Size of discriminator output (generated image)
g_output_size = 784
# Size of *first* hidden layer in the generator
g_hidden_size = 64

# instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

# Calculate losses
def real_loss(D_out, smooth=False):
    # compare logits to real labels
    # smooth labels if smooth=True
    batch_size = D_out.size(0)
    
    labels = torch.ones(batch_size)*0.9 if smooth else torch.ones(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    
    return loss

def fake_loss(D_out):
    # compare logits to fake labels
    batch_size = D_out.size(0)
    
    labels = torch.zeros(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    
    return loss
  
import torch.optim as optim

# learning rate for optimizers
lr = 0.002

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)
```



#### Training

```python
import pickle as pkl

# training hyperparams
num_epochs = 40

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
D.train()
G.train()
for epoch in range(num_epochs):
    
    for batch_i, (real_images, _) in enumerate(train_loader):
                
        batch_size = real_images.size(0)
        
        ## Important rescaling step ## 
        real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
        
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
        
        d_optimizer.zero_grad()
                
        # 1. Train with real images
        real_out = D(real_images)
        real_losses = real_loss(real_out, True)

        # Compute the discriminator losses on real images
        # use smoothed labels
        
        # 2. Train with fake images
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images        
        fake_out = D(fake_images)
        fake_losses = fake_loss(fake_out)
        
        # add up real and fake losses and perform backprop
        d_loss = real_losses + fake_losses
        d_loss.backward()
        d_optimizer.step()
        
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        
        g_optimizer.zero_grad()
        # 1. Train with fake images and flipped labels
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        g_out = D(fake_images)
        
        # perform backprop
        g_loss = real_loss(g_out)
        g_loss.backward()
        g_optimizer.step()        

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))
    
    # generate and save sample, fake images
    G.eval() # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to train mode


# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
```

## Deep Convolutional GANs(DCGANs)

Deep Convolutional GANs(DCGANS) are GANs. However, DCGANs uses convolutional networks for generator and discriminator.

- Discriminator of DCGANs

![image](https://user-images.githubusercontent.com/8471958/97831524-2613ec00-1c85-11eb-9243-99e9859c0d31.png)

- Generator of DCGANs

![image](https://user-images.githubusercontent.com/8471958/97831529-290edc80-1c85-11eb-86d1-f0d5436e7e1e.png)

### Code Samples

#### Define Network

##### Discriminator

```python
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)
    
class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()
        
        self.conv_dim = conv_dim

        # complete init function
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, batch_norm=False)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=4, stride=2, padding=1, batch_norm=True)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=4, stride=2, padding=1, batch_norm=True)
        
        self.fc = nn.Linear(4*4*conv_dim*4, 1)

    def forward(self, x):
        # complete forward function
        
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, 4*4*self.conv_dim*4)
        
        x = self.fc(x)
        
        return x    
```

##### Generator

```python
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    ## TODO: Complete this function
    ## create a sequence of transpose + optional batch norm layers
    layers = []
    deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    
    layers.append(deconv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    return nn.Sequential(*layers)

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        
        self.conv_dim = conv_dim

        # complete init function
        self.fc = nn.Linear(z_size, 4*4*conv_dim*4)
        
        self.deconv1 = deconv(in_channels=conv_dim*4, out_channels=conv_dim*2, kernel_size=4, stride=2, padding=1, batch_norm=True)
        self.deconv2 = deconv(in_channels=conv_dim*2, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, batch_norm=True)
        self.deconv3 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=4, stride=2, padding=1, batch_norm=False)
        

    def forward(self, x):
        # complete forward function
        
        x = self.fc(x)
        x = x.view(-1, conv_dim*4, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        
        return x
```

##### Building Network

```python
# define hyperparams
conv_dim = 32
z_size = 100

# define discriminator and generator
D = Discriminator(conv_dim)
G = Generator(z_size=z_size, conv_dim=conv_dim)

def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss
    
    import torch.optim as optim

# params
lr = 0.0002
beta1=0.5
beta2=0.999

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
```

#### Training

```python
import pickle as pkl

# training hyperparams
num_epochs = 30

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 300

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
for epoch in range(num_epochs):
    
    for batch_i, (real_images, _) in enumerate(train_loader):
                
        batch_size = real_images.size(0)
        
        # important rescaling step
        real_images = scale(real_images)
        
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
        
        d_optimizer.zero_grad()
        
        # 1. Train with real images

        # Compute the discriminator losses on real images 
        if train_on_gpu:
            real_images = real_images.cuda()
        
        D_real = D(real_images)
        d_real_loss = real_loss(D_real)
        
        # 2. Train with fake images
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        # move x to GPU, if available
        if train_on_gpu:
            z = z.cuda()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images            
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)
        
        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        g_optimizer.zero_grad()
        
        # 1. Train with fake images and flipped labels
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        if train_on_gpu:
            z = z.cuda()
        fake_images = G(z)
        
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake) # use real loss to flip labels
        
        # perform backprop
        g_loss.backward()
        g_optimizer.step()

        # Print some loss stats
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            losses.append((d_loss.item(), g_loss.item()))
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##    
    # generate and save sample, fake images
    G.eval() # for generating samples
    if train_on_gpu:
        fixed_z = fixed_z.cuda()
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train() # back to training mode


# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
```

## Conditional GANs(CGANs)

Conditional GANs(CGANs) is an extension of the GANs. The differences from GANs are that this architecture uses the label data as input and can predict the label of the image.

![image](https://user-images.githubusercontent.com/8471958/97831897-34aed300-1c86-11eb-8e4d-348a76a1b973.png)

The most important advantage of CGANs is that we can generate fake data for a specific label.

### Code Samples

Most codes are the same as the original GANs. However, the discriminator and generator are added embedding layer to apply labels. And the loss functions compare with actual labels.

#### Define Network

```python
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # For metadata, add embedding layer
        self.label_embedding = nn.Embedding(10, 10)
        
        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):       
        # flatten image
        x = x.view(-1, 28*28)
        
        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = self.dropout(x)
        x = self.fc4(x)

        return x

class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        # For label, add embedding layer
        self.label_embedding = nn.Embedding(10, 10)
        
        # define all layers
        self.fc1 = nn.Linear(input_size + 10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
        
        self.dropout = nn.Dropout(0.3)        

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        
        # pass x through all layers
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = self.dropout(x)
        x = self.fc4(x)        
        # final layer should have tanh applied
        x = torch.tanh(x)
        
        return x
      
# Calculate losses
def real_loss(D_out, labels, smooth=False):
    # compare logits to real labels
    # smooth labels if smooth=True
    criterion = nn.CrossEntropyLoss()
    loss = criterion(D_out.squeeze(), labels)
    
    return loss

def fake_loss(D_out):
    # compare logits to fake labels
    batch_size = D_out.size(0)
    
    labels = torch.ones(batch_size, dtype=int) * 10
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(D_out.squeeze(), labels)
    
    return loss
```

#### Training

```python
import pickle as pkl

# training hyperparams
num_epochs = 100

# keep track of loss and generated, "fake" samples
samples = []
sample_labels = []
losses = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()
fixed_l = torch.randint(0, 10, (sample_size,))

# train the network
D.train()
G.train()
for epoch in range(num_epochs):
    
    for batch_i, (real_images, real_labels) in enumerate(train_loader):
                
        batch_size = real_images.size(0)
        
        ## Important rescaling step ## 
        real_images = scale(real_images)  # rescale input images from [0,1) to [-1, 1)
        if train_on_gpu:
            real_images = real_images.cuda()
            real_labels = real_labels.cuda()
        
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
        
        d_optimizer.zero_grad()
                
        # 1. Train with real images
        real_out = D(real_images)
        real_losses = real_loss(real_out, real_labels, True)

        # Compute the discriminator losses on real images
        # use smoothed labels
        
        # 2. Train with fake images
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_labels = torch.randint(0, 10, (batch_size,))
        
        if train_on_gpu:
            z = z.cuda()
            fake_labels = fake_labels.cuda()
            
        fake_images = G(z, fake_labels)
        
        # Compute the discriminator losses on fake images        
        fake_out = D(fake_images)
        fake_losses = fake_loss(fake_out)
        
        # add up real and fake losses and perform backprop
        d_loss = real_losses + fake_losses
        d_loss.backward()
        d_optimizer.step()
        
        
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        
        g_optimizer.zero_grad()
        # 1. Train with fake images and flipped labels
        
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        
        # Fake labels
        fake_labels = torch.randint(0, 10, (batch_size,))
        
        if train_on_gpu:
            z = z.cuda()
            fake_labels = fake_labels.cuda()
            
        fake_images = G(z, fake_labels)
        
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        g_out = D(fake_images)
        
        # perform backprop
        g_loss = real_loss(g_out, fake_labels)
        g_loss.backward()
        g_optimizer.step()        

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))

    
    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))
    
    # generate and save sample, fake images
    G.eval() # eval mode for generating samples
    if train_on_gpu:
        fixed_z = fixed_z.cuda()
        fixed_l = fixed_l.cuda()
        
    samples_z = G(fixed_z, fixed_l)
    samples.append(samples_z)
    sample_labels.append(fixed_l)
    G.train() # back to train mode


# Save training generator samples
with open('train_samples_cgan_mc.pkl', 'wb') as f:
    pkl.dump(samples, f)
    
with open('train_samples_cgan_mc_label.pkl', 'wb') as f:
    pkl.dump(sample_labels, f)    
```

## CycleGAN

When a CycleGAN trains, and sees one batch of real images from set $X$ and $Y$, it trains by performing the following steps:

**Training the Discriminators**
1. Compute the discriminator $D_X$ loss on real images
2. Generate fake images that look like domain $X$ based on real images in domain $Y$
3. Compute the fake loss for $D_X$
4. Compute the total loss and perform backpropagation and $D_X$ optimization
5. Repeat steps 1-4 only with $D_Y$ and your domains switched!


**Training the Generators**
1. Generate fake images that look like domain $X$ based on real images in domain $Y$
2. Compute the generator loss based on how $D_X$ responds to fake $X$
3. Generate *reconstructed* $\hat{Y}$ images based on the fake $X$ images generated in step 1
4. Compute the cycle consistency loss by comparing the reconstructions with real $Y$ images
5. Repeat steps 1-4 only swapping domains
6. Add up all the generator and reconstruction losses and perform backpropagation + optimization

### Code Samples

#### Define Network

##### Discriminator

```python
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
    
class Discriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        
        self.conv5 = conv(conv_dim*8, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # define feedforward behavior
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.conv5(x)
        
        return x
```

##### Generator

```python
# residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs  
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        self.conv1 = conv(conv_dim, conv_dim, kernel_size=3, stride=1)
        self.conv2 = conv(conv_dim, conv_dim, kernel_size=3, stride=1)
        
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        
        out = F.relu(self.conv1(x))
        out = self.conv2(x) + x
        
        return out

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
    
class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)

        # 2. Define the resnet part of the generator
        self.res_blocks = nn.Sequential(*[ResidualBlock(conv_dim*4) for i in range(n_res_blocks)])

        # 3. Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4) 
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv3 = deconv(conv_dim, 3, 4)

    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.res_blocks(x)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))

        return x    
```

##### Building Network

```python
def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""
    
    # Instantiate generators
    G_XtoY = CycleGenerator(g_conv_dim, n_res_blocks)
    G_YtoX = CycleGenerator(g_conv_dim, n_res_blocks)
    # Instantiate discriminators
    D_X = Discriminator(d_conv_dim)
    D_Y = Discriminator(d_conv_dim)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y
  
def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    # how close is the produced output from being "fake"?
    return torch.mean((D_out-0)**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss
    recon_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    # return weighted loss
    return recon_loss * lambda_weight
  
import torch.optim as optim

# hyperparams for Adam optimizers
lr=0.0002
beta1=0.5
beta2= 0.999

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])
```

#### Training

```python
# train the network
def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, 
                  n_epochs=1000):
    
    print_every=10
    
    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    fixed_X = scale(fixed_X) # make sure to scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs+1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##
        d_x_optimizer.zero_grad()
        d_y_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        d_out_x = D_X(images_X)
        
        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 3. Compute the fake loss for D_X
        f_out_x = D_X(fake_X)
        
        # 4. Compute the total loss and perform backprop
        d_x_loss = real_mse_loss(d_out_x) + fake_mse_loss(f_out_x)
        d_x_loss.backward()
        d_x_optimizer.step()
        
        ##   Second: D_Y, real and fake loss components   ##
        
        d_out_y = D_Y(images_Y)
        fake_Y = G_XtoY(images_X)
        f_out_y = D_Y(fake_Y)
        
        d_y_loss = real_mse_loss(d_out_y) + fake_mse_loss(f_out_y)
        d_y_loss.backward()
        d_y_optimizer.step()        


        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()
        
        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        g_YtoX_loss = real_mse_loss(D_X(fake_X))

        # 3. Create a reconstructed y
        recon_Y = G_XtoY(fake_X)
        
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        recon_y_loss = cycle_consistency_loss(images_Y, recon_Y, lambda_weight=10)

        ##    Second: generate fake Y images and reconstructed X images    ##
        fake_Y = G_XtoY(images_X)
        g_XtoY_loss = real_mse_loss(D_Y(fake_Y))
        recon_X = G_YtoX(fake_Y)
        recon_x_loss = cycle_consistency_loss(images_X, recon_X, lambda_weight=10)
        

        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = g_XtoY_loss + g_YtoX_loss + recon_x_loss + recon_y_loss
        g_total_loss.backward()
        g_optimizer.step()
        
        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

            
        sample_every=100
        # Save the generated samples
        if epoch % sample_every == 0:
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16)
            G_YtoX.train()
            G_XtoY.train()

        # uncomment these lines, if you want to save your model
#         checkpoint_every=1000
#         # Save the model parameters
#         if epoch % checkpoint_every == 0:
#             checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)

    return losses
```

## [[Project] Face Generation Workspace](https://github.com/madigun697/udacity-nanodegree/tree/master/Deep%20Learning%20Nano%20Degree/5.%20Generative%20Adversarial%20Networks/Project%205.%20Generate%20faces)

In this project, you'll use generative adversarial networks to generate new images of faces.