import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

class LineDefect(Dataset):
	def __init__(self, x_array, y_labels, saved_encoder, cls_token_id, max_size=None):
		"""
		Args: ---------CHANGE PATHS PLEASE!!!!---------
		array (string): Path to the concatenated tokenized array (array of token-ids)
		y_labels (string): Path to the y_labels
		saved_encoder (string): Path to the saved encoder (saved by preprocess.py). We need this so that strings can be converted to tokens & vice versa
		"""
		with open(x_array, 'rb') as handle:
			self.x_array = pickle.load(handle)           
		with open(y_labels, 'rb') as handle:
			self.y_labels = pickle.load(handle)       
		with open(saved_encoder, 'rb') as handle:
			self.saved_encoder = pickle.load(handle)
		self.cls_token_id = cls_token_id

		if max_size:
			self.x_array = self.x_array[:max_size]
			self.y_labels = self.y_labels[:max_size]

	def __len__(self):
		return (len(self.x_array))
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		l = self.x_array[idx].tolist() # self.x_array[idx] is the idx-th example in dataset
		start_token = self.saved_encoder.encode('<START>')[0]
		end_token = self.saved_encoder.encode('<END>')[0]               
		# cls_token_id = 218799
		l.insert(0, self.cls_token_id) # Prepending <CLS> token_id to each datapoint       
		start_index = l.index(start_token)
		end_index = l.index(end_token)    
		#         Attention mask calculation -- CLS token also gets 1
		mask = np.zeros(len(l), dtype=int) 
		for index in range (len(l)):
			if l[index]!=0:
				mask[index]=1

#         Position id calculation -- CLS till <START> = 0 to m; <START>+1 till <END> = 0 to n; <END>+1 till 1000 = 0 to p
#         Token type calculation -- CLS-<START> = 0; <START>-<END> = 1; <END>-1000 = 2
		position = np.zeros(len(l), dtype=int)
		token_type_list = np.zeros(len(l), dtype=int)
		count=0
		token_type=0
		for index in range (0, start_index+1):
			position[index]=count
			token_type_list[index]=token_type
			count+=1    
		count=0
		token_type=1
		for index in range (start_index+1, end_index+1):
			position[index]=count
			token_type_list[index]=token_type
			count+=1           
		count=0
		token_type=2
		for index in range (end_index+1, len(position)):
			position[index]=count
			token_type_list[index]=token_type
			count+=1       
		l = torch.tensor(l)
		position = torch.tensor(position)
		token_type_list = torch.tensor(token_type_list)
		return l, position, token_type_list, mask, self.y_labels[idx]

