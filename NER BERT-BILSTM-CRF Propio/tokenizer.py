import pickle
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def token_train():

	# Load CSV
	df = pd.read_csv('APTNER_train.csv')
	df = df.dropna()
	#df = df.iloc[::100]

	# Define the input sentences and labels

	sentences = df["text"].values.tolist()
	labels = df["label"].values.tolist()
	etiquetas = list()
	for label in labels:
		etiquetas.append(list(map(int, label.replace("[","").replace("]","").split(", "))))

	# Tokenize the sentences
	tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

	# Convert the tokenized sentences to input IDs
	input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]

	# Pad the input IDs and create attention masks
	padded_input_ids = torch.tensor([ids + [0]*(max_len-len(ids)) for ids in input_ids])
	attention_masks = torch.tensor([[float(i > 0) for i in ids] for ids in padded_input_ids])   

	# Convert the labels to PyTorch tensors
	etiquetas = [torch.tensor(l) for l in etiquetas]

	# Pad the labels to the maximum length
	etiquetas = [torch.cat((l, torch.zeros(max_len - len(l), dtype=torch.int))) for l in etiquetas]

	etiquetas = torch.stack(etiquetas, dim=0)

	print(padded_input_ids.shape)
	print(attention_masks.shape)
	print(etiquetas.shape)

	padded_input_ids = padded_input_ids.reshape(107342, 504)
	print(padded_input_ids.shape)
	attention_masks = attention_masks.reshape(107342, 504)
	print(attention_masks.shape)
	etiquetas = etiquetas.reshape(107342, 504)
	print(etiquetas.shape)

	dataset = torch.utils.data.TensorDataset(padded_input_ids, attention_masks, etiquetas)


	with open('dataset_train2.pkl', 'wb') as f:
	    pickle.dump(dataset, f)
	print("Dataset guardado.")

def token_test():
	

	df_test = pd.read_csv('/Users/Carmen/Desktop/Laboral/teleco/Sobre TFGs/2023/Alejandro Lozano/nuestra_red/APTNER-main/APTNER_test.csv')
	df_test = df_test.dropna()

	# Define the input sentences and labels

	sentences_test = df_test["text"].values.tolist()
	labels_test = df_test["label"].values.tolist()
	etiquetas_test = list()
	for label in labels_test:
	    etiquetas_test.append(list(map(int, label.replace("[","").replace("]","").split(", "))))

	# Tokenize the sentences
	tokenized_texts_test = [tokenizer.tokenize(sent) for sent in sentences_test]

	# Convert the tokenized sentences to input IDs
	input_ids_test = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts_test]

	# Pad the input IDs and create attention masks
	padded_input_ids_test = torch.tensor([ids + [0]*(max_len -len(ids)) for ids in input_ids_test])
	attention_masks_test = torch.tensor([[float(i > 0) for i in ids] for ids in padded_input_ids_test])   

	# Convert the labels to PyTorch tensors
	etiquetas_test = [torch.tensor(l) for l in etiquetas_test]

	# Pad the labels to the maximum length
	etiquetas_test = [torch.cat((l, torch.zeros(max_len - len(l), dtype=torch.int))) for l in etiquetas_test]

	etiquetas_test = torch.stack(etiquetas_test, dim=0)

	print(padded_input_ids_test.shape)
	print(attention_masks_test.shape)
	print(etiquetas_test.shape)

	padded_input_ids_test = padded_input_ids_test.reshape(32424, 382)
	print(padded_input_ids_test.shape)
	attention_masks_test = attention_masks_test.reshape(32424, 382)
	print(attention_masks_test.shape)
	etiquetas_test = etiquetas_test.reshape(32424, 382)
	print(etiquetas_test.shape)

	dataset_test = torch.utils.data.TensorDataset(padded_input_ids_test, attention_masks_test, etiquetas_test)

	with open('dataset_test.pkl', 'wb') as f:
	    pickle.dump(dataset_test, f)
	print("Dataset guardado.")


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

max_len = 8022

token_train()
#token_test()
