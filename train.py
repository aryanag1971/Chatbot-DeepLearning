#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')


# In[14]:


import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()


# In[16]:


def tokenize(sentence):
  return nltk.word_tokenize(sentence)
def stem(word):
  return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence,all_words):
    tokenized_sentence=[stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag
sentence=["how","are","you"]
all_word=["what","are","you","doing","hel","how"]
    
     
 


# In[18]:


import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


# In[19]:


with open('intents.json','r') as f:
  intents=json.load(f)
all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
  tag=intent['tag']
  tags.append(tag)
  for pattern in intent['patterns']:
    w=tokenize(pattern)
    all_words.extend(w)
    xy.append((w,tag))
ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words)) 


# In[49]:


x_train=[]
y_train=[]
for (pattern,tag) in xy:
    bag=bag_of_words(pattern,all_words)
    x_train.append(bag)
    y_train.append(tags.index(tag))
    
x_train=np.array(x_train)
y_train=np.array(y_train,dtype=np.int64)


# In[50]:


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples
batch_size=8
dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)


# In[58]:


class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(NeuralNet,self).__init__()
        
        self.fc1=nn.Linear(input_size,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,output_size);
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.relu(self.fc1(x))
        out=self.relu(self.fc2(out))
        out=self.fc3(out)
        return out


# In[59]:


hidden_size=16 
output_size=len(tags)
input_size=len(all_words)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device)


# In[60]:


# create loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


# In[63]:


num_epochs=1000
for epoch in range(num_epochs):
    regular_loss=0
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)
        logit=model(words)
        loss=criterion(logit,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        regular_loss+=loss.item()
#     if(epoch%100==0):
#         print("Train_loss",regular_loss)


        


# In[64]:


# save the data
data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags
}
FILE="data.pth"
torch.save(data,FILE)


# In[65]:


import random


# In[ ]:


model.eval()
bot_name="Arya"
print("let's chat! type 'quit' to exit")
while True:
    sentence=input('You: ')
    if(sentence=="quit"):
        break
    sentence=tokenize(sentence)
    x=bag_of_words(sentence,all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x)
    output=model(x)
    a,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    for intent in intents["intents"]:
        if tag==intent["tag"]:
            print(f"{bot_name}:{random.choice(intent['responses'])}")
    
    


# In[67]:





# In[ ]:




