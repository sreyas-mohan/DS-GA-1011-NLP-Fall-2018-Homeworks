import glob
import os
import spacy
import string
from nltk import ngrams
from collections import Counter
import pickle

punctuations = string.punctuation
PAD_IDX = 0
UNK_IDX = 1

def read_all_files_folder(main_path, foldername, return_ratings=False):
	file_list = glob.glob(os.path.join(main_path, foldername, "*.txt"))

	corpus = [];
	if(return_ratings):
		ratings = [];

	for file_path in file_list:
		if(return_ratings):
			base = os.path.basename(file_path)
			ratings.append( int(os.path.splitext(base)[0].split('_')[1]))
	
		with open(file_path) as f_input:
			corpus.append(f_input.read())

	if(return_ratings):
		return corpus, ratings
	else:
		y_vec = [int(foldername == 'pos')]*len(corpus)
		return corpus, y_vec  


def load_data(train_path, test_path, return_ratings = False):
	x_train_pos, y_train_pos = read_all_files_folder(train_path, 'pos', return_ratings);
	x_train_neg, y_train_neg = read_all_files_folder(train_path, 'neg', return_ratings);
	x_train = x_train_pos + x_train_neg;
	y_train = y_train_pos + y_train_neg;
	
	x_test_pos, y_test_pos = read_all_files_folder(test_path, 'pos', return_ratings);
	x_test_neg, y_test_neg = read_all_files_folder(test_path, 'neg', return_ratings);
	x_test = x_test_pos + x_test_neg;
	y_test = y_test_pos + y_test_neg;
	
	return x_train, y_train, x_test, y_test

def tokenize_for_one_n(token_array, n):
	ngram_list =  ngrams(token_array, n);
	return [ ' '.join(grams) for grams in ngram_list]

def tokenize(sent, tokenizer, n=1, all_until_n = True):
	tokens = tokenizer(sent)
	token_array =  [token.text.lower() for token in tokens if (token.text not in punctuations)];
	if(n >1 and all_until_n):
		final_list = [];
		for i in range(1, n+1):
			final_list += tokenize_for_one_n(token_array, i);
	else:
		final_list =  ngrams(token_array, n);
		final_list = [ ' '.join(grams) for grams in final_list];
	return final_list


def tokenize_dataset(dataset, tokenizer, n, all_until_n = True):
	token_dataset = []
	all_tokens = []
	
	for sample in dataset:
		tokens = tokenize(sample, tokenizer, n, all_until_n)
		token_dataset.append(tokens)
		all_tokens += tokens

	return token_dataset, all_tokens


def load_tokenized_data(processed_path, type_str, x_val, y_val, tokenizer, n, token_scheme, all_until_n = True):
	val_file_name = '_'.join([type_str, token_scheme, str(n)])+'.p'
	full_file_path = os.path.join(processed_path, val_file_name) ;
	
	if(type_str == 'tokens'):
		
		if os.path.exists(full_file_path):
			token_list = pickle.load( open(full_file_path, "rb") )
		else:
			print('Tokenizing Train');
			_, token_list = tokenize_dataset(x_val, tokenizer, n, all_until_n);
			pickle.dump(token_list, open(full_file_path, "wb"))
			
		return token_list
			
	else:
		
		if os.path.exists(full_file_path):
			val_data = pickle.load( open(full_file_path, "rb") )
		else:
			print('Tokenizing '+type_str);
			val_data_tokens, _ = tokenize_dataset(x_val, tokenizer, n, all_until_n);
			val_data = {'x': val_data_tokens, 'y':y_val};
			pickle.dump(val_data, open(full_file_path, "wb"))
			
		return val_data['x'], val_data['y']


def build_vocab(all_tokens, max_vocab_size):
	# Returns:
	# id2token: list of tokens, where id2token[i] returns token that corresponds to token i
	# token2id: dictionary where keys represent tokens and corresponding values represent indices
	token_counter = Counter(all_tokens)
	vocab, count = zip(*token_counter.most_common(max_vocab_size))
	id2token = list(vocab)
	token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
	id2token = ['<pad>', '<unk>'] + id2token
	token2id['<pad>'] = PAD_IDX 
	token2id['<unk>'] = UNK_IDX
	return token2id, id2token

def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data




