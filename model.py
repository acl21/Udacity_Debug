import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, caption_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_size
        # Reviewer Note: I have explained this in the comment but this will help - https://discuss.pytorch.org/t/could-someone-explain-batch-first-true-in-lstm/15402
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)

        # the linear layer that maps the hidden state output dimension 
        # to the number of captions we want as output, vocab_size
        self.hidden2caption = nn.Linear(hidden_size, vocab_size)
        
        self.caption_size = caption_size
        
        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros(1, self.caption_size, self.hidden_size),
                torch.zeros(1, self.caption_size, self.hidden_size))
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        print(features.size())
        print(captions.size())
        #features_long=features.long()
        # Reviewer Note: There is an elegant way to do what you need with features
        features = features.unsqueeze(1)
        #embeds = self.word_embeddings(captions)
        embeds = self.word_embeddings(captions[:,:-1])
        print(embeds.shape)
        # Reviewer Note: () are healthier than []. Always ;)
        stacks=torch.cat((features, embeds), dim=1)
        print(stacks.shape)        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        # Reviewer Note: You have already unsqueezed your features, set batch_first=True so the concatenation is perfect.
        # No need to make hidden a class object. It is unused. However, in sample method you will do it iterantively so there you pass
        # the hidden state to the lstm. Here we are doing it all in one go. I hope that is clear. :) 
        lstm_out, lstm_hidden = self.lstm(stacks)
        
        # get the scores for the most likely tag for a word
        caption_outputs = self.hidden2caption(lstm_out)
        # Reviewer Note: You DO NOT have to do this since your loss function handles this. If you read the 
        # PyTorch documentation you will notice this - "The input is expected to contain raw, unnormalized scores for each class."
        #caption_scores = F.log_softmax(caption_outputs, dim=1)
        
        return caption_output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
    
