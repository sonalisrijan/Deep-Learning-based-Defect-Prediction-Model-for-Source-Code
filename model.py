import torch.nn as nn
from transformers import BertModel, AutoConfig

class LineBertModel(nn.Module):
	def __init__(self, vocab_size, dropout_prob, layers, num_labels=1):
		super(LineBertModel, self).__init__()
		# Using configs for uncased base BERT
		config = AutoConfig.from_pretrained("bert-base-cased")
		config.vocab_size = vocab_size
		config.max_position_embeddings = 1001 #1000 + 1 for CLS token
		config.type_vocab_size = 3
		config.num_hidden_layers = layers
		config.num_attention_heads = 4
		config.hidden_size = 256
		config.intermediate_size = 1024
		config.hidden_dropout_prob = dropout_prob

		# EMBEDDING: Training my own embedding
		self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size)

		# ENCODER: Training my own BERT encoder
		self.bert = BertModel(config) # BertModel class 
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.sigmoid = nn.Sigmoid()

	def forward(self, xarray, position, token_type_list, mask):

		#Convert the input index to embedding
		embeddings = self.word_embeddings(xarray)

		#Input embedding, position id, token type id and mask idx to the bert model
		outputs = self.bert(inputs_embeds=embeddings,
							position_ids=position,
							token_type_ids=token_type_list,
							attention_mask = mask)

		#Pass output of first token to neural network for classification prediction
		pooled_output = outputs[1] # pooled_output embedding of [CLS] token
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		return self.sigmoid(logits)
