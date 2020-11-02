# 4. Recurrent Neural Networks

Recurrent Neural Networks(RNNs) overcome that previous neural networks were trained the current inputs only. RNNs use the memory concept so that it can use previous inputs and outputs. This is useful in some domains like Natural Language Processing(NLP) or Time-series prediction.

[toc]

**Official course description**

*In this part, you’ll learn about Recurrent Neural Networks (RNNs) — a type of network architecture particularly well suited to data that forms  sequences like text, music, and time series data. You'll build a  recurrent neural network that can generate new text character by  character.*

*![img](https://video.udacity-data.com/topher/2018/September/5b96e4fc_screen-shot-2018-09-10-at-2.41.04-pm/screen-shot-2018-09-10-at-2.41.04-pm.png)*

*Examples of input/output sequence types.*

***Natural Language Processing***

*Then, you'll learn about word embeddings and implement the [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) model, a network that can learn about semantic relationships between  words. These are used to increase the efficiency of networks when you're processing text.*

*You'll combine embeddings and an RNN to predict the sentiment of movie  reviews, an example of common tasks in natural language processing.*

*In the **third project**, you'll use what you've learned here to generate new TV scripts from provided, existing scripts.* 

*![img](https://video.udacity-data.com/topher/2018/September/5b96e67c_screen-shot-2018-09-10-at-2.47.28-pm/screen-shot-2018-09-10-at-2.47.28-pm.png)*

*An example RNN structure in which an encoder represents the question: "how are you?" and a decoder generates the answer: "I am good"*

## Recurrent Neural Networks(RNNs)

The most important point in RNN is the state. The state is the result of the previous step. RNN uses not only the input values and current weight but also the state as input values. Because of that, the network refers to the previous step's output.

![image](https://user-images.githubusercontent.com/8471958/97823013-7e8abf80-1c6c-11eb-975c-d0236be57a94.png)
![image](https://user-images.githubusercontent.com/8471958/97823025-8e0a0880-1c6c-11eb-8c4a-e529b3ba031b.png)

### Code Samples

#### Define Network

```python
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        
        self.hidden_dim=hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  
        
        # get final output 
        output = self.fc(r_out)
        
        return output, hidden
```

#### Training

```python
# train the RNN
def train(rnn, n_steps, print_every):
    
    # initialize the hidden state
    hidden = None      
    
    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data 
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length + 1)
        data = np.sin(time_steps)
        data.resize((seq_length + 1, 1)) # input_size=1

        x = data[:-1]
        y = data[1:]
        
        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i%print_every == 0:        
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.') # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.') # predictions
            plt.show()
    
    return rnn
```

## Long Short-Term Memory Cells(LSTM)

RNN uses only the previous state. It means that if there are multiple subjects in the input data set, and the subject of input is switched, RNN cannot get a response properly. LSTM overcomes this shortage using two different memory, long-term and short-term.

### Structure of LSTM cell

![image](https://user-images.githubusercontent.com/8471958/95925175-f9e40a00-0d6d-11eb-9129-bc057363ec99.png)

![image](https://user-images.githubusercontent.com/8471958/95925205-0ff1ca80-0d6e-11eb-8514-a5a0a1319fb2.png)

#### Learn Gate

![image](https://user-images.githubusercontent.com/8471958/95925264-31eb4d00-0d6e-11eb-809e-6d6c69183e4c.png)

- Learn gate determines what memory has to learn or has to ignore in the short-term memory
- The previous short-term memory is combined with the current event
- The ignore factor(<img src="https://render.githubusercontent.com/render/math?math=i_t">) is calculated by previous short-term memory(<img src="https://render.githubusercontent.com/render/math?math=STM_{t-1}">) and current event(<img src="https://render.githubusercontent.com/render/math?math=E_t">)

#### Forget Gate

![image](https://user-images.githubusercontent.com/8471958/95925288-416a9600-0d6e-11eb-9641-7209b9246034.png)

- Forget gate determines what memory has to remain or has to forget in the long-term memory
- The forget factor(<img src="https://render.githubusercontent.com/render/math?math=f_t">) is calculated by previous short-term memory(<img src="https://render.githubusercontent.com/render/math?math=STM_{t-1}">) and current event(<img src="https://render.githubusercontent.com/render/math?math=E_t">)

#### Remember Gate

![image](https://user-images.githubusercontent.com/8471958/95925316-56dfc000-0d6e-11eb-9879-dad4156e6d19.png)

- Remember gate determines the next long-term memory to add up results of forget gate and learn gate together
- The result of forget gate = <img src="https://render.githubusercontent.com/render/math?math=LTM_{t-1} \cdot f_t = LTM_{t-1} \cdot \sigma(W_f[STM_{t-1}, E_t] + b_f)">
- The result of learn gate = <img src="https://render.githubusercontent.com/render/math?math=N_i \cdot i_t = tanh(W_n[STM_{t-1}, E_t] + b_n) \cdot \sigma(W_i[STM_{t-1}, E_t]+b_i)">
- The result of remember gate<img src="https://render.githubusercontent.com/render/math?math=(LTM_t)"> = <img src="https://render.githubusercontent.com/render/math?math=LTM_{t-1} \cdot f_t + N_i \cdot i_t">

#### Use Gate

![image](https://user-images.githubusercontent.com/8471958/95925326-65c67280-0d6e-11eb-86d1-5fa74c66bd2d.png)

- Use gate determines the next short-term memory.

### Code Sample

#### Define Network

```python
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # define all layers
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        
        batch_size = x.size(0)
        
        x = self.emb(x)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.fc(x)
        sig_out = self.sigmoid(x)
        
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
      
# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 400 
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)      
```

#### Training

```python
# loss and optimization functions
lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
```

#### Testing

```python
# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output, h = net(inputs, h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
```

## [[Project] Generate TV Scripts](https://github.com/madigun697/udacity-nanodegree/tree/master/Deep%20Learning%20Nano%20Degree/4.%20Recurrent%20Neural%20Networks/Project%204.%20Generate%20TV%20Scripts)

In this project, you'll generate your own Seinfeld TV scripts using  RNNs. You'll be using a Seinfeld dataset of scripts from 9 seasons. The  Neural Network you'll build will generate a new, "fake" TV script.