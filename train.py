import argparse
import pickle
import os
from os.path import join
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
import time
import torch
from torch.nn import BCELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import LineDefect
from model import LineBertModel

def train(args, model, train_dataset, eval_dataset):
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
	eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=8)

	loss_fct = BCELoss()
	optimizer = AdamW(model.parameters(), lr=args.lr)

	print("***** Running training *****")
	print("  Num examples = %d" % (len(train_dataset)))
	print("  Num Val examples = %d" % (len(eval_dataset)))
	print("  Num Epochs = %d" % (args.epochs))
	print("  Batch Size = %d" % (args.batch_size))

	output_dir = join(args.out, args.save)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	log_file = open(join(output_dir, 'log'),'w')

	global_step = 0
	best_val_auc = 0.0
	running_loss = 0.0
	model.zero_grad()
	for epoch in range(args.epochs):
		for step, batch in enumerate(train_dataloader):
			model.train()
			# start_time = time.time()
			xarray, position, token_type_list, mask, ylabel = batch
			xarray = xarray.to(args.device)
			position = position.to(args.device)
			token_type_list = token_type_list.to(args.device)
			mask = mask.to(args.device)
			ylabel = ylabel.to(args.device)
			# batch_end_time = time.time()

			output = model(xarray, position, token_type_list, mask)
			# output_time = time.time()
			loss = loss_fct(output.view(-1).to(torch.float32), ylabel.view(-1).to(torch.float32))
			
			loss.backward()
			optimizer.step()
			model.zero_grad()
			# loss_time = time.time()
			
			running_loss += loss.item()
			global_step += 1

			# print("Batch time",batch_end_time - start_time, "output_time", output_time - batch_end_time, "loss_time", loss_time - output_time)

			# print every logging_step steps
			if global_step % args.logging_step == 0 and global_step != 0:
				eval_result =  eval(args, model, eval_dataloader)
				print('Epoch: %d, Global Step: %d, Loss: %.3f, Eval Loss: %.3f, Eval F1score: %.3f, Eval AUC: %.3f' % (epoch + 1, global_step, (running_loss / args.logging_step) , eval_result['loss'], eval_result['f1'], eval_result['auc']))
				log_file.write('Epoch: %d, Global Step: %d, Loss: %.3f, Eval Loss: %.3f, Eval F1score: %.3f, Eval AUC: %.3f \n' % (epoch + 1, global_step, (running_loss / args.logging_step) , eval_result['loss'], eval_result['f1'], eval_result['auc']))
				running_loss = 0.0

				#If eval accuracy increases, save the model
				if eval_result['auc'] > best_val_auc:
					best_val_auc = eval_result['auc']
					torch.save(model.state_dict(),os.path.join(output_dir, "model_state_dict.pt"),)
					torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

def eval(args, model, eval_dataloader):
	eval_loss = 0.0
	eval_steps = 0
	eval_ylabels = []
	eval_outputs = []

	loss_fct = BCELoss()
	model.eval()
	for step, batch in enumerate(eval_dataloader):
		xarray, position, token_type_list, mask, ylabel = batch
		xarray = xarray.to(args.device)
		position = position.to(args.device)
		token_type_list = token_type_list.to(args.device)
		mask = mask.to(args.device)
		ylabel = ylabel.to(args.device)

		with torch.no_grad():
			output = model(xarray, position, token_type_list, mask)
			loss = loss_fct(output.view(-1).to(torch.float32), ylabel.view(-1).to(torch.float32))
			eval_loss += loss.item()
			eval_ylabels += ylabel.cpu().detach().tolist()
			eval_outputs += output.view(-1).cpu().detach().tolist()
		eval_steps += 1

	eval_loss = eval_loss / eval_steps

	eval_ylabels = np.asarray(eval_ylabels)
	eval_outputs = np.asarray(eval_outputs)
	fpr, tpr, _ = roc_curve(eval_ylabels, eval_outputs)
	roc_auc = auc(fpr, tpr)

	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")

	output_dir = join(args.out, args.save)
	figname = join(output_dir, args.save + "_auc_model.png")
	plt.savefig(figname, dpi=300)


	#Conver the output to 1 and 0 to compute f1 score
	eval_outputs[eval_outputs >= 0.5] = 1
	eval_outputs[eval_outputs < 0.5] = 0
	f1 = f1_score(eval_ylabels, eval_outputs)

	result = {"f1": f1, "loss": eval_loss, "auc": roc_auc}
	return result

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
	parser.add_argument("--epochs", default=15, type=int, help="num of epochs")
	parser.add_argument("--dropout", default=0.1, type=float, help="dropout probability")
	parser.add_argument("--batch_size", default=16, type=int, help="batch size")
	parser.add_argument("--logging_step", default=250, type=int, help="Step size for logging")
	parser.add_argument("--device_id", type=int, default=0, help="cude device id")
	parser.add_argument("--layers", type=int, default=6, help="transformer layers")
	parser.add_argument("--out",default='./trained_model',type=str,help="Directory to save the model")
	parser.add_argument("--save",default='test',type=str,help="Model name to save")
	args = parser.parse_args()

	#Set CUDA device (or CPU)
	device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
	args.device = device

	#Tokenizer
	# with open('post_encoder_class.pickle', 'rb') as handle:
	with open('200k_post_encoder_class.pickle', 'rb') as handle: #for 200k data
		saved_encoder = pickle.load(handle)

	cls_token_id = saved_encoder.vocab_size
	vocab_size = saved_encoder.vocab_size + 1 #Add 1 for CLS token

	Initialize the mode
	model = LineBertModel(vocab_size=vocab_size, dropout_prob=args.dropout, layers=args.layers)
	model.to(args.device)

	#Load the dataset
	# 100k data
	# train_dataset = LineDefect(x_array='./clean_post_x_train.pickle', y_labels='clean_post_y_train.pickle', saved_encoder='post_encoder_class.pickle', cls_token_id=cls_token_id)
	# val_dataset = LineDefect(x_array='./post_x_valid.pickle', y_labels='post_y_valid.pickle', saved_encoder='post_encoder_class.pickle', cls_token_id=cls_token_id)
	# test_dataset = LineDefect(x_array='./post_x_test.pickle', y_labels='post_y_test.pickle', saved_encoder='post_encoder_class.pickle', cls_token_id=cls_token_id)

	# #200K data
	train_dataset = LineDefect(x_array='./clean_200k_post_x_train.pickle', y_labels='clean_200k_post_y_train.pickle', saved_encoder='200k_post_encoder_class.pickle', cls_token_id=cls_token_id)
	val_dataset = LineDefect(x_array='./200k_post_x_valid.pickle', y_labels='200k_post_y_valid.pickle', saved_encoder='200k_post_encoder_class.pickle', cls_token_id=cls_token_id)
	test_dataset = LineDefect(x_array='./200k_post_x_test.pickle', y_labels='200k_post_y_test.pickle', saved_encoder='200k_post_encoder_class.pickle', cls_token_id=cls_token_id)

	train(args, model, train_dataset, eval_dataset=val_dataset)

	#Load the best model and run it on test set
	best_model = LineBertModel(vocab_size=vocab_size, dropout_prob=args.dropout, layers=args.layers)
	best_model.load_state_dict(torch.load(os.path.join(args.out, args.save + "/model_state_dict.pt")))
	best_model.to(args.device)
	best_model.eval()
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)
	test_result = eval(args, best_model, test_dataloader)
	print(test_result)
	
if __name__ == '__main__':
	main()
	
