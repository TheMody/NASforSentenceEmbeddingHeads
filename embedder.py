from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from transformers.utils import logging
from transformers import glue_convert_examples_to_features
logging.set_verbosity_error()
#import nvidia_smi

models = ['bert-base-uncased',
         'google/electra-small-discriminator',
         'distilbert-base-uncased',
         'gpt2',
         'bag_of_words'
          ]

class Bag_of_words(nn.Module):
    def __init__(self, pad_token):
        super(Bag_of_words, self).__init__()
        self.length = 30522
        self.pad_token = pad_token
        return

    def __call__(self, input_ids, **kwargs):
        output = []
        for sentence in input_ids:
            bag = [0]*self.length
            for token in sentence:
                # done when first padding token occurs
                if token == self.pad_token:
                    break
                bag[token] += 1
            output.append(bag)
        output = torch.FloatTensor(output)
        return output


class NLP_embedder(nn.Module):

    def __init__(self, hyperparameters, num_classes, batch_size):
        super(NLP_embedder, self).__init__()
        self.type = 'nn'
        self.batch_size = batch_size
        self.hyperparameters = hyperparameters
        self.padding = True
        self.bag = False
        self.lasthiddenstate = 0
        if hyperparameters["model_name"] == models[0]:
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            output_length = 768
        elif hyperparameters["model_name"] == models[1]:
            from transformers import ElectraTokenizer, ElectraModel
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            self.model = ElectraModel.from_pretrained('google/electra-small-discriminator')
            output_length = 256
        elif hyperparameters["model_name"] == models[2]:
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.lasthiddenstate = -1 
            output_length = 768
        elif hyperparameters["model_name"] == models[3]:
            from transformers import GPT2Tokenizer, GPT2Model
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2Model.from_pretrained('gpt2')
            self.padding = False
            self.lasthiddenstate = -1  # needed because gpt2 always outputs the same token in the beginning
            self.batch_size = 1  # has to be 1 since gpt-2 does not have padding tokens
            output_length = 768
        elif hyperparameters["model_name"] == models[4]:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = Bag_of_words(0)
            self.bag = True
            output_length = 30522
            

        else:
            print("model not supported", hyperparameters["model_name"])
        self.fc1 = nn.Linear(output_length, hyperparameters["hidden_fc"])
        self.fc2 = nn.Linear(hyperparameters["hidden_fc"], num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hyperparameters["lr"])  # higher lrs than 1e-5 result in no change in loss
        self.softmax = torch.nn.Softmax(dim=1)
        
#         test = self.tokenizer(["hallo ","sadsa das ist wahnsinn"], return_tensors="pt", padding=True)
#         print(self(test))
#         print(self.embed(test))
        
    def forward(self, x):
        x = self.model(**x)
        #x = torch.mean(x.last_hidden_state, dim = 1)
        if self.bag:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = x.to(device)
        else:
            x = x.last_hidden_state[:, self.lasthiddenstate]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x
    
    def embed(self, x):
    
        resultx = None

        if self.bag:
            tokens = self.tokenizer(x, return_tensors="pt", padding=True)
            resultx = self.model(**tokens)
            resultx = resultx.detach()
            return resultx

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)

            batch_x = self.model(**batch_x)
#             number_notzeros = 0
#             for encoding in batch_x:
#                 for number in encoding:
#                     if not number == 0:
#                         number_notzeros += 1
#             print(number_notzeros)
            if not self.bag:
                batch_x = batch_x.last_hidden_state[:, self.lasthiddenstate]

            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))
        
        return resultx
     
    def classify(self, x):
        if self.bag:  # in this case the model is not yet on gpu
            # keep in mind: bag-of-words embeddings might be too large to fit on the gpu all at once
            # -> batch-wise classification might be necessary
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x
     
    def fit(self, x, y, epochs=1):

        for e in range(epochs):
            for i in range(math.ceil(len(x) / self.batch_size)):
              #  batch_x, batch_y = next(iter(data))
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
           #     batch_x = glue_convert_examples_to_features(, tokenizer, max_length=128,  task=task_name)
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                if not self.bag:
                    batch_x = batch_x.to(device)
                y_pred = self(batch_x)
                loss = self.criterion(y_pred, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # if i % 100 == 0:
                #     # check available memory
                #     print("batch", i)
                #     nvidia_smi.nvmlInit()
                #     handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                #     info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                #     #            print("Total memory:", info.total)
                #     print("Free memory:", info.free)
                #     #            print("Used memory:", info.used)
                #     nvidia_smi.nvmlShutdown()


                #print(len(x))
#                 if i % np.max((1,int((len(x)/self.batch_size)*0.01))) == 0:
#                     print(i, loss.item())
                # print(y_pred, batch_y)

                if not self.bag:
                    # remove data from gpu
                    batch_x = batch_x.to('cpu')
                    batch_y = batch_y.to('cpu')
                    y_pred = y_pred.to('cpu')
                    del batch_x
                    del batch_y
                    del y_pred

        torch.cuda.empty_cache()

        return 
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not self.bag:
                batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

            if device == 'cuda':
                batch_x = batch_x.to('cpu')
                del batch_x

        if device == 'cuda':
            resultx = resultx.to('cpu')
        return resultx
    
    
class hybrid_classifier():
    
    def __init__(self, hyperparameters, num_classes, batch_size):
        from sklearn import svm
        self.type = 'svm'
        self.embedder = NLP_embedder(hyperparameters, num_classes, batch_size)
        self.classifier = svm.SVC()
        
    def to(self, device):
        self.embedder = self.embedder.to(device)
        return self
        
    def fit(self, x, y, epochs=1):
        embedded_X = self.embedder.embed(x).to("cpu")
        self.classifier.fit(embedded_X, y)
        
    def predict(self, x):
        embedded_X = self.embedder.embed(x).to("cpu")
        return self.classifier.predict(embedded_X)
    
    def embed(self, x):
        return self.embedder.embed(x)
    
    def classify(self, x):
        x = x.to("cpu")
        return self.classifier.predict(x)
        
        

