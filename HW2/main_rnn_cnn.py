
# coding: utf-8

# In[ ]:


data_path = './hw2_data/'


# In[ ]:


# First lets improve libraries that we are going to be used in this lab session
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
import pandas as pd
import io
import sys
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
random.seed(134)

parser = argparse.ArgumentParser(description="CNN or RNN")
parser.add_argument("--model_type", type=str, default='cnn', help="CNN or RNN")
opt = parser.parse_args()
print(opt);

# In[ ]:


MAX_VOCAB_SIZE = 150000
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

MAX_SENTENCE_LENGTH = 50
BATCH_SIZE = 128
EMBED_SIZE = 300


# In[ ]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ### Dataset Preparation

# In[ ]:


# y_label_map = {'neutral':0,
#                 'contradiction': 1, 
#                 'entailment': 2};

y_label_map = {'contradiction':0,'neutral':1,'entailment':2}

def get_string_tokenized_data(data_mat_path):
    df = pd.read_csv(os.path.join(data_mat_path), sep="\t")
    data = np.array(df);
    data = data.astype(str)
    
    tokenized_data_x = len(data) * [None];
    y_labels = [y_label_map[x] for x in data[:, 2] ];
    
    all_tokens = [];
    
    for i,x in enumerate(data):
        tokenized_data_x[i] = [x[0].split(), x[1].split()];
        all_tokens += (tokenized_data_x[i][0] + tokenized_data_x[i][1])

    return all_tokens, tokenized_data_x, y_labels
        


# convert token to id in the dataset
def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens1, tokens2 in tokens_data:
        index_list1 = [token2id[token] if token in token2id else UNK_IDX for token in tokens1]
        index_list2 = [token2id[token] if token in token2id else UNK_IDX for token in tokens2]
        indices_data.append([index_list1, index_list2])
    return indices_data


def build_vocab(all_tokens):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(MAX_VOCAB_SIZE))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embedding_dict = np.random.randn(MAX_VOCAB_SIZE+2, EMBED_SIZE)
    all_train_tokens = []
    i = 0
    
    for line in fin:
        tokens = line.rstrip().split(' ')
        all_train_tokens.append(tokens[0])
        embedding_dict[i+2] = list(map(float, tokens[1:]))
        i += 1
        if i == MAX_VOCAB_SIZE:
            break
            
    return embedding_dict, all_train_tokens


# In[ ]:


_, val_data_x, val_data_y = get_string_tokenized_data(os.path.join(data_path, 'snli_val.tsv'))
_, train_data_x, train_data_y = get_string_tokenized_data(os.path.join(data_path, 'snli_train.tsv'))

fasttext_embedding_dict, all_fasttext_tokens = load_vectors('wiki-news-300d-1M.vec')

token2id, id2token = build_vocab(all_fasttext_tokens)
train_data_indices = token2index_dataset(train_data_x, token2id)
val_data_indices = token2index_dataset(val_data_x, token2id)


# double checking
print ("Train dataset size is {}".format(len(train_data_indices)))
print ("Val dataset size is {}".format(len(val_data_indices)))


# In[ ]:


count = 0;
for x in train_data_indices:
    if 1 in set(x[0]):
        count+=1


# In[ ]:


count/len(train_data_indices)


# ### Dataset Pytorch Classes

# In[ ]:


class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_x, target_list):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets

        """
        self.data_x = data_x;
        self.target_list = target_list
        
        assert(len(data_x) == len(target_list))

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        prem_token_idx = self.data_x[key][0][:MAX_SENTENCE_LENGTH]
        hyp_token_idx = self.data_x[key][1][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [prem_token_idx, hyp_token_idx, label]


def encode_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    prem_data_list = []
    hyp_data_list = []
    label_list = []
    length_list = []
    # print("collate batch: ", batch[0][0])
    # batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
    # padding
    for datum in batch:
        prem_padded_vec = np.pad(np.array(datum[0]),
                                 pad_width=((0, MAX_SENTENCE_LENGTH - len(datum[0]))),
                                 mode="constant", constant_values=0)
        hyp_padded_vec = np.pad(np.array(datum[1]),
                                pad_width=((0, MAX_SENTENCE_LENGTH - len(datum[1]))),
                                mode="constant", constant_values=0)
        prem_data_list.append(prem_padded_vec)
        hyp_data_list.append(hyp_padded_vec)
    return [torch.from_numpy((np.array(prem_data_list))), torch.from_numpy(np.array(hyp_data_list)),
            torch.LongTensor(label_list)]


# In[ ]:


train_dataset = SNLIDataset(train_data_indices, train_data_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=encode_collate_func,
                                           shuffle=True)

val_dataset = SNLIDataset(val_data_indices, val_data_y)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=encode_collate_func,
                                           shuffle=True)


# ## Neural Networks

# #### Generic Functions to be used

# In[ ]:


# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for prem_data, hyp_data, labels in loader:
        prem_data_batch, hyp_data_batch, label_batch = prem_data.to(device), hyp_data.to(device),labels.to(device)
        outputs = F.softmax(model(prem_data_batch, hyp_data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
    return (100 * correct / total)


def plot_acc(train_accs, val_accs, filename):
    f = plt.figure()
    plt.plot(train_accs, label='train');
    plt.plot(val_accs, label='val');
    plt.title(filename);
    plt.legend()

    f.savefig(os.path.join(filename[:3], filename + ".pdf"), bbox_inches='tight')
    plt.show()


# In[ ]:


def train_and_evaluate(model_type = 'cnn', kernel_size = 3, hidden_size = 3, linear_hid_dim = 10, 
                       combine_method = 'concat', regularization = 'none'):

    file_name = '_'.join([model_type, 'kernel_size='+str(kernel_size), 'hidden_size='+str(hidden_size), 'linear_hid_dim='+str(linear_hid_dim), 
                       'combine_method='+str(combine_method), 'regularization='+str(regularization)]);
    print('\n'.join([model_type, 'kernel_size='+str(kernel_size), 'hidden_size='+str(hidden_size), 'linear_hid_dim='+str(linear_hid_dim), 
                       'combine_method='+str(combine_method), 'regularization='+str(regularization)]))
    sys.stdout.flush()
    
    learning_rate = 1e-3;
    num_epochs = 10;
    
    dropout = (regularization == 'dropout')
    

    if(model_type == 'cnn'):
        model = CNN(EMBED_SIZE , hidden_size, MAX_VOCAB_SIZE+2, kernel_size, linear_hid_dim, combine_method, dropout);
    elif(model_type == 'rnn'):
        model = RNN(EMBED_SIZE , hidden_size, MAX_VOCAB_SIZE+2,  linear_hid_dim, combine_method, dropout);
    else:
        error('invalid model type')
        
    model = model.to(device);

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if regularization == 'weight_decay':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = 0;
    best_state_dict = None;
    
    train_acc_array = [];
    val_acc_array = [];
    
    for epoch in range(num_epochs):

        for i, (prem, hyp, label) in enumerate(train_loader):
            
#             if i>300:
#                 break;
            model.train()
            
            prem_batch, hyp_batch, label_batch = prem.to(device), hyp.to(device), label.to(device);
            
            optimizer.zero_grad()
            
            outputs = model(prem_batch, hyp_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            # validate every 300 iterations
            if (i+1) % 300 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                
                if val_acc > best_val:
                    best_state_dict = model.state_dict();
                    best_val = val_acc;
                    
                val_acc_array.append(val_acc);
                train_acc = test_model(train_loader, model);
                train_acc_array.append(train_acc)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Train Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc, train_acc))
                sys.stdout.flush()
                
    plot_acc(train_acc_array, val_acc_array, file_name)
    
    print ("After training for {} epochs".format(num_epochs))
    print ("Train Acc {}".format(test_model(train_loader, model)))
    print ("Val Acc {}".format(test_model(val_loader, model)))
    sys.stdout.flush()
    
    model.load_state_dict(best_state_dict)
    
    return test_model(train_loader, model), test_model(val_loader, model), model


# ## CNN Model

# In[ ]:


class CNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, kernel_size, linear_hidden_dim, combine_method, dropout):

        super(CNN, self).__init__()
        
        assert(kernel_size % 2 == 1);
        assert(combine_method in ['concat', 'mul', 'add']);
        
        padding = int( (kernel_size-1)/2 );        

        self.hidden_size = hidden_size;
        self.combine_method = combine_method;
        self.dropout = dropout;
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding.from_pretrained(torch.from_numpy(np.array(fasttext_embedding_dict)).cuda(), freeze = False)
#         self.embedding.weight.data.copy_(torch.from_numpy(np.array(fasttext_embedding_dict).copy()))
    
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding)


        
        if combine_method == 'concat':
            self.linear_layers1 = nn.Linear(hidden_size*2, linear_hidden_dim)
        else:
            self.linear_layers1 = nn.Linear(hidden_size, linear_hidden_dim)
            
        self.linear_layers2 =  nn.Linear(linear_hidden_dim, 3)
        
#         self.xavier_init(self.linear_layers1);
#         self.xavier_init(self.linear_layers2);
        
        if self.dropout:
            self.dropout_layer = nn.Dropout(0.5);
            
        
    def xavier_init(self, layer):
        torch.nn.init.xavier_normal_(layer.weight.data)
        layer.bias.data.fill_(0.01)
    
    def indivual_encoding(self, x):
        batch_size, seq_len = x.size()

        embed = self.embedding(x)
        m = (x == 1)
        m = m.unsqueeze(2).repeat(1, 1, EMBED_SIZE).type(torch.FloatTensor).to(device);
       
        embed = m * embed + (1-m) * embed.clone().detach()
        
        hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))

        hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        
#         print(hidden.shape)
        hidden = torch.max(hidden, 1)[0]
#         print(hidden.shape)
        
        return hidden
    
    def forward(self, prem, hyp):
        prem_vector = self.indivual_encoding(prem);
        hyp_vector = self.indivual_encoding(hyp);
        
        if self.combine_method == 'concat':
            final_code = torch.cat((prem_vector, hyp_vector), dim=1);
        elif self.combine_method == 'mul':
            final_code = prem_vector * hyp_vector;
        elif self.combine_method == 'add':
            final_code = prem_vector + hyp_vector;
            

        final_code = self.linear_layers1(final_code);
        final_code = F.relu(final_code);
        if self.dropout:
            final_code = self.dropout_layer(final_code);
        final_code = self.linear_layers2(final_code)
            
            
        return final_code

        


# ## RNN Model

# In[ ]:


class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, linear_hidden_dim, combine_method, dropout):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.combine_method = combine_method;
        self.dropout = dropout;
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(np.array(fasttext_embedding_dict).copy()))
        
        self.bi_gru = nn.GRU(emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        
        if combine_method == 'concat':
            self.linear_layers1 = nn.Linear(hidden_size*2, linear_hidden_dim)
        else:
            self.linear_layers1 = nn.Linear(hidden_size, linear_hidden_dim)
            
        self.linear_layers2 =  nn.Linear(linear_hidden_dim, 3)
        
        if self.dropout:
            self.dropout_layer = nn.Dropout(0.5);
            
    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(2, batch_size, self.hidden_size).to(device)
        return hidden
    
    def encode(self, x):
        
        batch_size, seq_len = x.size()
        self.hidden = self.init_hidden(batch_size)
        embed = self.embedding(x)
        m = (x == 1)
        m = m.unsqueeze(2).repeat(1, 1, EMBED_SIZE).type(torch.FloatTensor).to(device)
        embed = m * embed + (1-m) * embed.clone().detach()
        
        # embed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu().numpy(), batch_first=True)
        
        output, hidden = self.bi_gru(embed, self.hidden)
        hidden = torch.sum(hidden, dim = 0)
        
#         hidden = hidden.index_select(0, idx_unsort)
        
        return hidden
    
    
    def forward(self, prem, hyp):
        batch_size, seq_len = prem.size()

        prem_vector = self.encode(prem)
        hyp_vector = self.encode(hyp)
        
        if self.combine_method == 'concat':
            final_code = torch.cat((prem_vector, hyp_vector), dim=1);
        elif self.combine_method == 'mul':
            final_code = prem_vector * hyp_vector;
        elif self.combine_method == 'add':
            final_code = prem_vector + hyp_vector;
            

        final_code = self.linear_layers1(final_code);
        final_code = F.relu(final_code);
        if self.dropout:
            final_code = self.dropout_layer(final_code);
        final_code = self.linear_layers2(final_code)
            
            
        return final_code


# ### Hyperparameter Tuninig and Training

# In[ ]:


model_type_list = [opt.model_type]
hidden_size_list = [50, 150]
kernel_size_list = [3, 5]
linear_hid_dim_list = [400]
regularization_list = ['none', 'dropout'];
# regularization_list = ['none']
combine_method_list = ['concat', 'mul', 'add']
# combine_method_list = ['concat']


# In[ ]:


assert(len(model_type_list) == 1)


# In[ ]:


best_acc = 0;


# In[ ]:


with open(os.path.join(model_type_list[0], 'accuracy.txt'), 'w') as thefile:
    for model_type in model_type_list:
        for hidden_size in hidden_size_list:
            for linear_hid_dim in linear_hid_dim_list:
                for regularization in regularization_list:
                    for combine_method in combine_method_list:
                    
                                if(model_type == 'cnn'):
                                    for kernel_size in kernel_size_list:
                                        train_acc, val_acc, model = train_and_evaluate(model_type, kernel_size , hidden_size, linear_hid_dim, 
                                                                           combine_method, regularization);
                                        
                                        thefile.write(','.join([model_type, str(kernel_size), str(hidden_size), str(linear_hid_dim), 
                                                           str(combine_method), str(regularization), str(train_acc),
                                                               str(val_acc)])+'\n')
                                else:
                                        kernel_size = None;
                                        
                                        train_acc, val_acc, model = train_and_evaluate(model_type, kernel_size , hidden_size, linear_hid_dim, 
                                                                           combine_method, regularization);
                                        
                                        thefile.write(','.join([model_type,  str(hidden_size), str(linear_hid_dim), 
                                                           str(combine_method), str(regularization), str(train_acc),
                                                               str(val_acc)])+'\n')
                                        
                                if val_acc > best_acc:
                                    best_acc = val_acc;
                                    best_hidden_size, best_linear_hid_dim = hidden_size, linear_hid_dim
                                    best_kernel_size, best_combine_method = kernel_size, combine_method
                                    torch.save(model, os.path.join('.', model_type_list[0], 'best_'+model_type+'.pth') )
                                    


# In[ ]:


print(model_type_list)
print('\n'.join(['best_val_acc: '+ str(best_acc), 
                    'best_hidden_size: '+str(best_hidden_size), 'best_linear_hid_dim: '+str(best_linear_hid_dim),
                    'best_kernel_size: '+str(best_kernel_size), 'best_combine_method: '+combine_method]))

