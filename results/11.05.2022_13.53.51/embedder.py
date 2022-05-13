from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
         'gpt2'
          ]

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn =  nn.MultiheadAttention(input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out,_ = self.self_attn(x,x,x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class NLP_embedder(nn.Module):

#             hidden_fc=ag.space.Int(lower=5, upper=200),
#              number_layers = ag.space.Int(lower = 1, upper = 10),
#              lr=ag.space.Real(lower=5e-7, upper=1e-4, log=True),
#              CNNs = ag.space.Categorical(ag.space.Dict(hidden_fc=ag.space.Int(lower=5, upper=200),
#                                                        number_layers = ag.space.Int(lower = 1, upper = 5),
#                                                        kernel_size = ag.space.Int(lower = 3, upper = 11)
#                                                     #   pooling = ag.space.Categorical("max", "mean", "first")
#                  ),ag.space.Dict()
#              
#              ),
#              pooling = ag.space.Categorical("max", "mean", "[CLS]"),
#              layer_norm = ag.space.Categorical("layer_norm", "Batch_norm", "none"),
#              freeze_base = ag.space.Categorical(True,False),
#              Attention = ag.space.Categorical(ag.space.Dict(embed_dim=ag.space.Categorical(8,16,32,64,128,256,512),
#                                                             num_heads=ag.space.Categorical(1,2,4,8),
#                                                        number_layers = ag.space.Int(lower = 1, upper = 5)
#                  ),ag.space.Dict())
#             )

    def __init__(self,  num_classes, batch_size, args):
        super(NLP_embedder, self).__init__()
        self.type = 'nn'
        self.batch_size = batch_size
        self.padding = True
        self.bag = False
        self.num_classes = num_classes
        self.lasthiddenstate = 0
        self.args = args
      #  if hyperparameters["model_name"] == models[0]:
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.output_length = 768
        
#         from transformers import RobertaTokenizer, RobertaModel
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#         self.model = RobertaModel.from_pretrained('roberta-base')
#         self.output_length = 768
#         elif hyperparameters["model_name"] == models[1]:
#             from transformers import ElectraTokenizer, ElectraModel
#             self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
#             self.model = ElectraModel.from_pretrained('google/electra-small-discriminator')
#             output_length = 256
#         elif hyperparameters["model_name"] == models[2]:
#             from transformers import DistilBertTokenizer, DistilBertModel
#             self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#             self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
#             self.lasthiddenstate = -1 
#             output_length = 768
#         elif hyperparameters["model_name"] == models[3]:
#             from transformers import GPT2Tokenizer, GPT2Model
#             self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#             self.model = GPT2Model.from_pretrained('gpt2')
#             self.padding = False
#             self.lasthiddenstate = -1  # needed because gpt2 always outputs the same token in the beginning
#             self.batch_size = 1  # has to be 1 since gpt-2 does not have padding tokens
#             output_length = 768
            
#         if args.freeze_base:
#             for param in self.model.parameters():
#                 param.requires_grad = False
        self.construct_head(args)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.softmax = torch.nn.Softmax(dim=1)
        
        

    def construct_head(self,args):
        
        #attention
        if len(args.Attention.keys()) > 0:
           self.atts = nn.ModuleList([EncoderBlock(input_dim = self.output_length, num_heads = args.Attention["num_heads"], 
                        dim_feedforward  = self.output_length, dropout=0.2) for i in range(args.Attention["number_layers"])])
                                       
        
        ffc_input_size = self.output_length
        #cnns
        if len(args.CNNs.keys()) > 0:
            self.ccn1 = nn.Conv1d(self.output_length, args.CNNs["hidden_fc"], args.CNNs["kernel_size"], padding = int(args.CNNs["kernel_size"]/2))
            self.ccns = nn.ModuleList([nn.Conv1d(args.CNNs["hidden_fc"], args.CNNs["hidden_fc"], args.CNNs["kernel_size"], padding = int(args.CNNs["kernel_size"]/2))
                                            for i in range(args.CNNs["number_layers"]-1)])
            ffc_input_size = args.CNNs["hidden_fc"]
            
    #    LSTM
#         if len(self.args.LSTMs.keys()) > 0:
#             self.rnn = nn.LSTM(ffc_input_size,args.LSTMs["hidden_fc"] , args.LSTMs["number_layers"], batch_first = True)
#             ffc_input_size = args.LSTMs["hidden_fc"]
            
        #fully connected layers
        if args.number_layers >= 2:
            self.fc1 = nn.Linear(ffc_input_size,args.hidden_fc)
            self.fc2 = nn.Linear(args.hidden_fc, self.num_classes)
            self.fcs = nn.ModuleList([nn.Linear(args.hidden_fc, args.hidden_fc) for i in range(args.number_layers-2)])
        else:
            self.fc1 = nn.Linear(ffc_input_size,self.num_classes)
            
        
        
        
    def forward(self, x):
        x = self.model(**x)   
        
        x = x.last_hidden_state
        
        #Attention
        if len(self.args.Attention.keys()) > 0:
            for att in self.atts:
                x = att(x)
        
        #CNNs
        if len(self.args.CNNs.keys()) > 0:
            x = torch.transpose(x,1,2)
            x = F.relu(self.ccn1(x))
            if self.args.CNNs["skip"]:
                x_in = x
            for ccn in self.ccns:
                x =  F.relu(ccn(x))
            if self.args.CNNs["skip"]:
                x = x_in + x
            x = torch.transpose(x,1,2)
        
#         #LSTMs
#         if len(self.args.LSTMs.keys()) > 0:
#             x,_ = self.rnn(x)
        
        #pooling
        if self.args.pooling == "[CLS]":
            x = x[:, self.lasthiddenstate]
        if self.args.pooling == "max":
            x,_ = torch.max(x, 1)
        if self.args.pooling == "mean":
            x = torch.mean(x,1)
        
        
        
        #fully connected layers
        if self.args.number_layers >= 2:
            x = F.relu(self.fc1(x))
            for fc in self.fcs:
                x =  F.relu(fc(x))
            x = F.relu(self.fc2(x))
        else:
            x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    
     
    def fit(self, x, y, epochs=1, X_val= None,Y_val= None, reporter = None):
        
        self.scheduler = CosineWarmupScheduler(optimizer= self.optimizer, 
                                               warmup = math.ceil(len(x)*epochs *0.1 / self.batch_size) ,
                                                max_iters = math.ceil(len(x)*epochs  / self.batch_size))
        
        accuracy = None
        for e in range(epochs):
            start = time.time()
            for i in range(math.ceil(len(x) / self.batch_size)):
              #  batch_x, batch_y = next(iter(data))
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
           #     batch_x = glue_convert_examples_to_features(, tokenizer, max_length=128,  task=task_name)
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 196, truncation = True)
             #   print(batch_x["input_ids"].size())
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                y_pred = self(batch_x)
                loss = self.criterion(y_pred, batch_y)          
                loss.backward()
                self.optimizer.step()

#                 if i % np.max((1,int((len(x)/self.batch_size)*0.001))) == 0:
#                     print(i, loss.item())
               # print(y_pred, batch_y)

                self.scheduler.step()
#                 batch_x = batch_x.to('cpu')
#                 batch_y = batch_y.to('cpu')
#                 y_pred = y_pred.to('cpu')
#                 del batch_x
#                 del batch_y
#                 del y_pred
#                torch.cuda.empty_cache()
            if X_val != None:
                with torch.no_grad():
                    accuracy = self.evaluate(X_val, Y_val)
                    print("accuracy after", e, "epochs:", float(accuracy.cpu().numpy()), "time per epoch", time.time()-start)
                    if reporter != None:
                        reporter(objective=float(accuracy.cpu().numpy()), epoch=e+1)
                
                

        return float(accuracy.cpu().numpy())
    
    def evaluate(self, X,Y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = Y.to(device)
        y_pred = self.predict(X)
        accuracy = torch.sum(Y == y_pred)
        accuracy = accuracy/Y.shape[0]
        return accuracy
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 196, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return torch.argmax(resultx, dim = 1)
    
    

        
        

