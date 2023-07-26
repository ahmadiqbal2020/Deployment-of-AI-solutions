#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,re
from tqdm import tqdm
from sklearn.utils import class_weight
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.parallel import DataParallel


# Overview
# 1. Parse EMAIL using REGEXP
# 2. Remove trivial <html> in email body and other trivial punctiations
# 3. Then feeding the pre-processed text into BERT-LARGE 340millon parameter model.
# 4. USING PCA to reduce dimen after multiple expriments it was found it is best to reduce from 1024 to 768 dimen.
# 5. After multiple up-sampling and down-sampling experiments it was found unbalanced train data performed better.
# 6. Gaussian NAIVE BAYES and 0.90 was AUC

# 1. Automated Function for encoding text using BERT-LARGE MODEL
# - https://huggingface.co/bert-large-uncased
# - https://arxiv.org/abs/1810.04805
# - **USING MAX POOLING of BERT last layers because it perfoms better in rare text prediction**
# - It input text and gives 1024 dimen vector

# In[3]:


def encode_text(txt_comp, model_name, max_seq_length):
    #it tokenize bert for specific [CLS] and [SEP] token
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    encoded_texts = []
    #mps is apple hardware accelerator used for GPU acceleratrion
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    model = model.to(device)
    #model parallel is for parallel computation
    model_parallel = DataParallel(model)
    
    i=0
    for text in txt_comp:
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_seq_length)
        input_ids = torch.tensor([tokens]).to(device)
        with torch.no_grad():
            outputs = model_parallel(input_ids)
            last_layer_hidden_states = outputs.last_hidden_state
        # this performs max pooling of last layers for better detection of anomly detection
        pooled_output,_ = torch.max(last_layer_hidden_states, dim=1)
        encoded_texts.append(pooled_output)
        print(i,"th step completed")
        i+=1
    encoded_texts = torch.cat(encoded_texts, dim=0)
    
    return encoded_texts


# In[4]:


#this function stores all name for files in array. It is being used for BERT
folder_path = "spam/"  
file_list = os.listdir(folder_path)
spam_list=[]
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        spam_list.append(filename)


# In[5]:


#this function stores all name for files in array. It is being used for BERT
folder_path = "ham/"
file_list = os.listdir(folder_path)
ham_list=[]
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        ham_list.append(filename)


# In[6]:


print("no of spam",len(spam_list),"& no of ham",len(ham_list)


# 2. This precpessing function
# - remove html tags<>
# - remove trivial punctuations

# In[7]:


def prepocess_html(text):
    clean_text = re.sub(r"<.*?>", "", text)
    clean_text = clean_text.replace("&nbsp;", "")
    clean_text=re.sub(r"http\S+|www\S+", "", clean_text)
    clean_text=re.sub(r"=\w*\d+\b", "", clean_text)
    clean_text=re.sub(r"[\(\)\[\]]", "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text.rstrip())
    clean_text=re.sub(r"=", " ", clean_text)
    clean_text = re.sub(r"(\b\w)\s+(\w+\b)", r"\1\2", clean_text)
    clean_text = re.sub(r"([.?!])(\S)", r"\1 \2", clean_text)
    clean_text=re.sub(r'<.*?>', '', clean_text)
    clean_text=re.sub(r'\\', '', clean_text)
    clean_text=re.sub(r'a', '', clean_text)

    return clean_text


# 3. This precpessing function
# - parse EMAIL and extract relevant text
# 

# In[8]:


import email

import re

def parse_email(email):
    parsed_data = {}
    # this extracts the subject
    subject_match = re.search(r'Subject: (.+)', email)
    if subject_match:
        parsed_data['subject'] = subject_match.group(1)
    #this extracts the body    
    body_match = re.search(r'\n\n(.+)', email, re.DOTALL)
    if body_match:
        parsed_data['body'] = body_match.group(1)
    #concat both subject and body. At same time remove html from body using helper function 
    #which too works on regexp
    return parsed_data['subject']+prepocess_html(parsed_data['body'])


# 4. This precpessing function
# - use above helper function and stores text into array

# In[201]:


model_name = 'bert-large-uncased'
max_seq_length = 512
def spam_preprocess(dire,arr):
    txt=[]
    for i in tqdm(range(len(arr))):
        file_path = dire.format(arr[i])
        with open(file_path, 'r',encoding='utf-8', errors='ignore') as file:
            try:
                parsed=parse_email(file.read())
                txt.append(parsed)
            except:
                pass
    return txt


# In[13]:


spam_txt=spam_preprocess("spam/{}",spam_list)


# In[14]:


ham_txt=spam_preprocess("ham/{}",ham_list)


# In[15]:


print(len(spam_txt),len(ham_txt))


# 5. This code
# - use extracted relevant text and use BERT large to encode

# In[16]:


spam_body_enc=encode_text(spam_txt, model_name, max_seq_length)
ham_body_enc=encode_text(ham_txt, model_name, max_seq_length)


# In[18]:


spam_body_enc.shape,ham_body_enc.shape


# In[20]:


spam_body_enc[0]


# In[22]:


np.save("spam_data.npy",spam_body_enc.cpu())


# In[23]:


np.save("ham_data.npy",ham_body_enc.cpu())


# In[24]:


get_ipython().system('ls')


# In[28]:


np.load("ham_data.npy").shape


# In[29]:


np.load("spam_data.npy").shape


# In[36]:


np.save("y_spam_data",np.ones(1886))


# In[37]:


np.save("y_ham_data",np.zeros(6695))


# In[40]:


np.load("y_spam_data.npy").shape


# In[41]:


np.load("y_ham_data.npy").shape


# 6. Converting and stroing torch encoding then loading stored encoding again into numpy vector form

# In[3]:


X=np.vstack((np.load("ham_data.npy"),np.load("spam_data.npy")))
y=np.concatenate((np.load("y_ham_data.npy"), np.load("y_spam_data.npy")), axis=0)


# In[4]:


X.shape,y.shape


# In[5]:


X.shape

