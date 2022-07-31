# used a dictionary to represent an intents JSON file
data = {"intents": [
{"tag": "greeting",
"responses": ["Howdy Partner!", "Hello", "How are you doing?",   "Greetings!", "How do you do?"]},
{"tag": "age",
"responses": ["I am 25 years old", "I was born in 1998", "My birthday is July 3rd and I was born in 1998", "03/07/1998"]},
{"tag": "date",
"responses": ["I am available all week", "I don't have any plans",  "I am not busy"]},
{"tag": "name",
"responses": ["My name is James", "I'm James", "James"]},
{"tag": "goodbye",
"responses": ["It was nice speaking to you", "See you later", "Speak soon!"]}
]}

import numpy as np
import torch
import requests
import json
import re
from flask import Flask, redirect, url_for, request, render_template
from markupsafe import escape
import pickle
import pandas as pd
import random
import torch.nn as nn
import transformers
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


# specify GPU
device = torch.device("cpu")

import openpyxl
df = pd.read_excel("C:/Users/Alexei/Desktop/Project Stuff/BP/Bert Model/dialogs.xlsx")

uniq = df['Label'].nunique()

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
# check class distribution
df['Label'].value_counts(normalize = True)
# class MyCustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == "__main__":
#             module = "Bert_Arch"
#         return super().find_class(module, name)

# with open('Bert Model.pt', 'rb') as f:
#     unpickler = MyCustomUnpickler(f)
#     model = unpickler.load

# checkpoint = torch.load('Bert Model', map_location='cpu')
# model = checkpoint['model_state_dict']
train_text, train_labels = df['Text'], df['Label']

#Bert Model
from transformers import AutoModel, BertTokenizerFast
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

#Roberta Model
from transformers import RobertaTokenizer, RobertaModel
# Load the Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Import Roberta pretrained model
bert = RobertaModel.from_pretrained('roberta-base')

#DistillBert Model
from transformers import DistilBertTokenizer, DistilBertModel
# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

#Sample data for distillbert-base-uncased tokenizer
text = ["this is a distil bert model.","data is oil"]
# Encode the text
encoded_input = tokenizer(text, padding=True,truncation=True, return_tensors='pt')
#print(encoded_input)

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
        train_text.tolist(),
        max_length = 8,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#define a batch size
batch_size = 16
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Define model architecture
class BERT_Arch(nn.Module):
        def __init__(self, bert):
                super(BERT_Arch, self).__init__()
                self.bert = bert

# dropout layer
                self.dropout = nn.Dropout(0.2)

# relu activation function
                self.relu =  nn.ReLU()
# dense layer
                self.fc1 = nn.Linear(768,512)
                self.fc2 = nn.Linear(512,256)
                self.fc3 = nn.Linear(256,uniq)
#softmax activation function
                self.softmax = nn.LogSoftmax(dim=1)
#define the forward pass
        def forward(self, sent_id, mask):
                #pass the inputs to the model
                cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
                x = self.fc1(cls_hs)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.dropout(x)
# output layer
                x = self.fc3(x)
# apply softmax activation
                x = self.softmax(x)
                return x


# freeze all th parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
        param.requires_grad = False
model = BERT_Arch(bert)
# push the model to GPU
model = model.to(device)
# from torchinfo import summary
# summary(model
#Optimizer
from transformers import AdamW
        # define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)
from sklearn.utils.class_weight import compute_class_weight
        #compute the class weights
class_wts = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels)
        #print(class_wts
#Balancing weights while calculating error
# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
        # loss function
cross_entropy = nn.NLLLoss(weight=weights)
        #Setting up epochs
from torch.optim.lr_scheduler import StepLR
        # empty lists to store training and validation loss of each epoch
train_losses=[]
        # number of training epochs
epochs = 200
        # We can also use learning rate scheduler to achieve better results
lr_sch = StepLR(optimizer, step_size=100, gamma=0.1)

                #Fine tune the model
                # function to train the model
def train():
        model.train()
        total_loss = 0
        # emptylist to save model predictions
        total_preds=[]
        # iterae over batches
        for step,batch in enumerate(train_dataloader):
        # progrss update after every 50 batches.
                if step % 50 == 0 and not step == 0:
                        print('  Batch {:>5,}  of  {:>5,}.'.format(step,    len(train_dataloader)))
        # push he batch to gpu
                        batch = [r.to(device) for r in batch]
                        sent_id, mask, labels = batch
        # get mdel predictions for the current batch
                        preds = model(sent_id, mask)
        # compue the loss between actual and predicted values
                        loss = cross_entropy(preds, labels)
        # add o to the total loss
                        total_loss = total_loss + loss.item()
        # backwrd pass to calculate the gradients
                        loss.backward()
        # clip he the gradients to 1.0. It helps in preventing the    exploding gradient problem
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # updat parameters
                        optimizer.step()
        # clearcalculated gradients
                        optimizer.zero_grad()
        # We ar not using learning rate scheduler as of now
        # lr_sc.step()
        # modelpredictions are stored on GPU. So, push it to CPU
                        preds=preds.detach().cpu().numpy()
        # appen the model predictions
                        total_preds.append(preds)
        # compue the training loss of the epoch
                        avg_loss = total_loss / len(train_dataloader)
        # preditions are in the form of (no. of batches, size of batch, no. of classes).
        # reshae the predictions in form of (number of samples, no. of classes)
                        total_preds  = np.concatenate(total_preds, axis=0)
        #return the loss and predictions
                        return avg_loss, total_preds
        #Model raining
for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        #trainde
        train_loss, _ = train()
        # appetrainin and validation loss
        train_losses.append(train_loss)
        # it cmake yor experiment reproducible, similar to set  random seed to all options where there needs a random seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
print(f'\nTraining Loss: {train_loss:.3f}')


def get_prediction(str):
        str = re.sub(r'[^a-zA-Z]+',"", str)
        test_text = [str]
        model.eval()
        tokens_test_data = tokenizer(
                test_text,
                max_length = 8,
                pad_to_max_length=True,
                truncation=True,
                return_token_type_ids=False
        )
        test_seq = torch.tensor(tokens_test_data['input_ids'])
        test_mask = torch.tensor(tokens_test_data['attention_mask'])
        preds = None
        with torch.no_grad():
                preds = model(test_seq.to(device), test_mask.to(device))
                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis = 1)
                print("Intent Identified:", le.inverse_transform(preds)[0])
                return le.inverse_transform(preds)[0]
def get_response(message):
        intent = get_prediction(message)
        for i in data['intents']:
                if i["tag"] == intent:
                        result = random.choice(i["responses"])
                        return result