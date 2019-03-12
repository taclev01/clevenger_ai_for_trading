import json
import nltk
import os
import random
import re
import torch

from torch import nn, optim
import torch.nn.functional as F

with open(os.path.join('..', '..', 'data', 'project_6_stocktwits', 'twits.json'), 'r') as f:
    twits = json.load(f)

"""print out the number of twits"""

# TODO Implement 
print(len(twits['data']))

messages = [twit['message_body'] for twit in twits['data']]
# Since the sentiment scores are discrete, we'll scale the sentiments to 0 to 4 for use in our network
sentiments = [twit['sentiment'] + 2 for twit in twits['data']]

nltk.download('wordnet')

def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - tokenize by splitting the string on whitespace 
        - removes any single character tokens
    
    Parameters
    ----------
        message : The text message to be preprocessed.
        
    Returns
    -------
        tokens: The preprocessed text into tokens.
    """ 
    #TODO: Implement 
    
    # Lowercase the twit message
    text = message.lower()
    
    # Replace URLs with a space in the message
    site_pattern = r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+'
    text = re.sub(site_pattern, ' ', text)
    
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    ticker_pattern = r'\$\w*\b'
    text = re.sub(ticker_pattern, ' ', text)
    
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    user_pattern = r'\@\w*\b'
    text = re.sub(user_pattern, ' ', text)

    # Replace everything not a letter with a space
    not_letter_pattern = r'[^A-Za-z]'
    text = re.sub(not_letter_pattern, ' ', text)
    
    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()

    # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
    wnl = nltk.stem.WordNetLemmatizer()
    # new_words = [w for w in words if w not in stopwords.words("english")]
    tokens = [wnl.lemmatize(token,pos='n') for token in tokens if len(token) > 1]
    tokens = [wnl.lemmatize(token,pos='v') for token in tokens if len(token) > 1]
    
    
    return tokens
    #return text


tokenized = [preprocess(messsage) for messsage in messages]

from collections import Counter


"""
Create a vocabulary by using Bag of words
"""
# TODO: Implement 
word_list = []
for twit in tokenized:
    for word in twit:
        word_list.append(word)

bow = Counter(word_list)

"""
Set the following variables:
    freqs
    low_cutoff
    high_cutoff
    K_most_common
"""

# TODO Implement 

# Dictionart that contains the Frequency of words appearing in messages.
# The key is the token and the value is the frequency of that word in the corpus.
tot_messages = len(tokenized)
freqs = {word:word_count/tot_messages for word,word_count in bow.items()}

#print(freqs['aposbeast'])
#print(freqs['the'])


# Float that is the frequency cutoff. Drop words with a frequency that is lower or equal to this number.
low_cutoff = 0.0000007

# Integer that is the cut off for most common words. Drop words that are the `high_cutoff` most common words.
high_cutoff = 10

# The k most common words in the corpus. Use `high_cutoff` as the k.
K_most_common_list = bow.most_common(high_cutoff)
K_most_common = [K_most_common_list[ii][0] for ii in range(high_cutoff)]


filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in K_most_common)]
#print(K_most_common)
#len(filtered_words) 

"""
Set the following variables:
    vocab
    id2vocab
    filtered
"""

#TODO Implement

# A dictionary for the `filtered_words`. The key is the word and value is an id that represents the word. 
# be sure to start the enumeration at 1 rather than the default 0
vocab = {word:word_id for word_id,word in enumerate(filtered_words,1)}
#print(vocab['the'])
#print('the' in vocab)
# Reverse of the `vocab` dictionary. The key is word id and value is the word. 
id2vocab = {ii:word for word,ii in vocab.items()}
#print(id2vocab[3])

# tokenized with the words not in `filtered_words` removed.
# these are your tokenized twits that we will be removing the filtered words from
filtered = [[word for word in message if word in vocab] for message in tokenized]
#for message in messages:
#    for word in message:
        

#filtered = id2vocab.keys()

prob_neut_drop = 0.4
print(0.5*prob_neut_drop)

balanced = {'messages': [], 'sentiments':[]}

n_neutral = sum(1 for each in sentiments if each == 2)
N_examples = len(sentiments)
keep_prob = (N_examples - n_neutral)/4/n_neutral

for idx, sentiment in enumerate(sentiments):
    message = filtered[idx]
    #print(message)
    if len(message) == 0:
        # skip this message because it has length zero
        continue
    elif sentiment != 2 or random.random() < keep_prob:
        balanced['messages'].append(message)
        balanced['sentiments'].append(sentiment) 


n_neutral = sum(1 for each in balanced['sentiments'] if each == 2)
N_examples = len(balanced['sentiments'])
n_neutral/N_examples

token_ids = [[vocab[word] for word in message] for message in balanced['messages']]
sentiments = balanced['sentiments']

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.
        
        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        
        super().__init__()
        # define the parameters to be used for the network:
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size # this is the "hidden" dimension of this network
        self.output_size = output_size # 5 output dimensions for the 5 different sentiment states
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        self.gpu_avail = torch.cuda.is_available()
        
        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(self.vocab_size,self.embed_size)
        
        # Setup additional layers
        self.lstm = nn.LSTM(self.embed_size, self.lstm_size, self.lstm_layers, dropout=self.dropout, batch_first=False)
        self.drop_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.lstm_size,self.output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def init_hidden(self, batch_size):
        """ 
        Initializes hidden state
        
        Parameters
        ----------
            batch_size : The size of batches.
        
        Returns
        -------
            hidden_state
            
        """
        
        # TODO Implement 
        
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        #print(self.gpu_avail)
        if (self.gpu_avail):
            hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda())
        else:
            hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                      weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                     weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        
        return hidden


    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.
        
        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """
        
        # TODO Implement 
        #batch_size = nn_input.size(0)
        
        nn_input = nn_input.long()
        
        embeds = self.embedding(nn_input)
        
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        
        # stack up lstm outputs
        lstm_out = lstm_out[-1,:,:]
        
        # dropout and fully-connected layer
        out = self.drop_layer(lstm_out)
        out = self.fc(out)
        
        # softmax layer:
        logps = self.log_softmax(out)
        
        return logps, hidden_state


model = TextClassifier(len(vocab), 10, 6, 5, dropout=0.1, lstm_layers=2)
model.embedding.weight.data.uniform_(-1, 1)
my_input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
hidden = model.init_hidden(4)

#print(model)
#print(my_input)
#print(hidden)

logps, _ = model.forward(my_input, hidden)
print(logps)

def dataloader(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    """ 
    Build a dataloader.
    """
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages)

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]
        
        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            # Left pad!
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[start_idx:, batch_num] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
        
        yield batch, label_tensor

"""
Split data into training and validation datasets. Use an appropriate split size.
The features are the `token_ids` and the labels are the `sentiments`.
"""   

# TODO Implement 
split_frac = 0.8
tot_toks = len(token_ids)
split_idx = int(tot_toks*split_frac)

train_features = token_ids[:split_idx]
train_labels = sentiments[:split_idx]
valid_features = token_ids[split_idx:]
valid_labels = sentiments[split_idx:]

print("id length:", len(token_ids))
print("label length:", len(sentiments))
print('train features:', len(train_features))
print('train labels:', len(train_labels))
print('valid features:', len(valid_features))
print('valid labels:', len(valid_labels))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
model.embedding.weight.data.uniform_(-1, 1)
model.to(device)

"""
Train your model with dropout. Make sure to clip your gradients.
Print the training loss, validation loss, and validation accuracy for every 100 steps.
"""

epochs = 2
batch_size = 512
learning_rate = 0.001
clip = 5
seq_length = 20

print_every = 100
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))
    
    steps = 0
    for text_batch, labels in dataloader(
            train_features, train_labels, batch_size=batch_size, sequence_length=seq_length, shuffle=True):
        steps += 1
        hidden = model.init_hidden(labels.shape[0])
        
        #print('Text batch:',text_batch.shape)
        #print('Labels:', labels.shape)
        
        # Set Device
        text_batch, labels = text_batch.to(device), labels.to(device)
        for each in hidden:
            each.to(device)
        
        # TODO Implement: Train Model
        
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        hidden = tuple([each.data for each in hidden])
        
        # zero accumulated gradients
        model.zero_grad()
        
        # get the output from the model
        output, hidden = model(text_batch, hidden)
        
        #print('Hidden:',hidden[1].shape)
        #print('Output:',output.shape)
        
        # calculate the loss and perform backprop
        loss = criterion(output, labels.long())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if steps % print_every == 0:
            
            val_losses = []
            
            model.eval()
            #import pdb; pdb.set_trace()
            # TODO Implement: Print metrics
            for valid_batch,valid_labels in dataloader(
                valid_features, valid_labels, batch_size=batch_size,sequence_length=seq_length, shuffle=True):
                # Get validation loss
                val_h = model.init_hidden(valid_labels.shape[0])
                #val_h = model.init_hidden(batch_size)
            
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                
                valid_batch,valid_labels = valid_batch.to(device), valid_labels.to(device)
                
                for each in val_h:
                    each.to(device)
                    
                val_out, val_h = model.forward(valid_batch,val_h)
                
                val_loss = criterion(val_out, valid_labels.long())
                
                val_losses.append(val_loss.item())
                
                # calculate metrics:
                ps = torch.exp(val_out)
                top_p,top_class = ps.topk(1,dim=1)
                
            
            model.train()
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(steps),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))






---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-28-363f9645dede> in <module>()
     62             # TODO Implement: Print metrics
     63             for valid_batch,valid_labels in dataloader(
---> 64                 valid_features, valid_labels, batch_size=batch_size,sequence_length=seq_length, shuffle=True):
     65                 # Get validation loss
     66                 val_h = model.init_hidden(valid_labels.shape[0])

<ipython-input-20-b1d8598928b4> in dataloader(messages, labels, sequence_length, batch_size, shuffle)
      7         random.shuffle(indices)
      8         messages = [messages[idx] for idx in indices]
----> 9         labels = [labels[idx] for idx in indices]
     10 
     11     total_sequences = len(messages)

<ipython-input-20-b1d8598928b4> in <listcomp>(.0)
      7         random.shuffle(indices)
      8         messages = [messages[idx] for idx in indices]
----> 9         labels = [labels[idx] for idx in indices]
     10 
     11     total_sequences = len(messages)

IndexError: index 199909 is out of bounds for dimension 0 with size 178


























