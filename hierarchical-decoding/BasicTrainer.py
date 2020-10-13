
from utils.vocab import Vocab
from utils.util import initialize_weights, set_seed, count_parameters, save_params, build_vocab
import utils.constants as constants
from utils.loss_new import LabelSmoothing, LossCompute
from utils.optimizer import NoamOpt

from Dataloader import ParallelDataset, get_dataloader
from Translate import translate

from models.Sequence2Sequence import Seq2seq
from layers.Encoder import EncoderRNN
from layers.Decoder import DecoderRNN
from layers.Decoder import Generator
from layers.Attention import BahdanauAttention
from layers.Embedding import Embedding

import torch
import torch.nn as nn

import math
import time



def build_dataset(source_files, target_files, batch_size, shuffle=False, \
			source_vocabs=None, target_vocabs=None, mtl=False, max_length=180):
	'''
		This method builds a dataset and dataloader for all tasks
		source_files: path for each source file (each file represents a task)
		target_files: path for each target file (each file represents a task)
		batch_size: the size of the batch
		shuffle: shuffle the dataset
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		mtl: if true an specific target vocabulary is used for each dataset sharing he source vocab, otherwise, all are built separately
		max_length: max length of the source/target lines
	'''

	loaders = []

	for index, (source_file, target_file) in enumerate(zip(source_files, target_files)):
		if mtl is True:
			_set = ParallelDataset(source_file, target_file, max_length = max_length, \
									source_vocab = source_vocabs[0], target_vocab = target_vocabs[index])
		else:
			_set = ParallelDataset(source_file, target_file, max_length = max_length, \
									source_vocab = source_vocabs[0], target_vocab = target_vocabs[0])

		loader = get_dataloader(_set, batch_size, shuffle=shuffle)
		loaders.append(loader)
	return loaders


def load_model(args, source_vocabs, target_vocabs, device, max_length):
	'''
		This method loads a pre-trained model
		args: arguments for loading the model
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		device: if use gpu or cpu
		max_length: max length of a sentence
	'''
	if args.load_encoder:
		from collections import OrderedDict
		encoder = OrderedDict()
		model = torch.load(args.model)
		for item in model:
			if item.startswith("encoder"):
				encoder[item.replace("encoder.","")] = model[item]
		print("Building an model using a pre-trained encoder ... ")
		current = build_model(args, source_vocabs, target_vocabs, device, max_length, encoder)
		return current
	else:
		mtl = build_model(args, source_vocabs, target_vocabs, device, max_length)
		mtl.load_state_dict(torch.load(args.model))
		print("Building an model using the encoder and the decoder ... ")
		return mtl


def build_model(args, source_vocabs, target_vocabs, device, max_length , encoder=None):
	'''
		This method builds a model from scratch or using the encoder of a pre-trained model
		args: arguments for loading the model
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		device: if use gpu or cpu
		max_length: max length of a sentence
		encoder: if the encoder is passed as a pre-trained model
	'''

	input_dim = source_vocabs[0].len()

	enc = EncoderRNN(args.embedding_size, 
		args.hidden_size, 
		args.encoder_layer, 
		args.encoder_dropout,
		args.layer_normalization,
		args.max_length).to(device)

	attention = BahdanauAttention(args.hidden_size)

	output_dim = target_vocabs[0].len()
	dec = DecoderRNN(args.embedding_size, 
			args.hidden_size, 
			attention, 
			args.decoder_layer,  
			args.decoder_dropout,
			attention,
			norm=args.layer_normalization).to(device)

	if args.tie_embeddings:
		model = Seq2seq(enc, dec, Embedding(input_dim, args.embedding_size, args.embedding_dropout, args.layer_normalization), 
			Embedding(output_dim, args.embedding_size, args.embedding_dropout, args.layer_normalization), 
			Generator(args.embedding_size, output_dim), True)
	else:
		model = Seq2seq(enc, dec, Embedding(input_dim, args.embedding_size, args.embedding_dropout, args.layer_normalization), 
			Embedding(output_dim, args.embedding_size, args.embedding_dropout, args.layer_normalization), 
			Generator(args.embedding_size, output_dim))

	model.apply(initialize_weights);
	model.to(device)

	return model


def train_step(model, loader, loss_compute, device, task_id = 0):
	'''
		This method performs training on a step (only one batch)
		model: the model being trained
		loader: dataloader that provides the batches
		loss_compute: function to compute the loss
		device: if use gpu or cpu
		task_id: task id that is being trained (0 as default)
	'''

	model.train()

	(src, tgt, src_mask, tgt_mask, src_lengths, tgt_lengths, tgt_pred) = next(iter(loader))

	src = src.to(device)
	tgt = tgt.to(device)
	src_mask = src_mask.to(device)
	tgt_mask = tgt_mask.to(device)
	tgt_pred = tgt_pred.to(device)

	out, _, pre_output = model.forward(src, tgt,
								src_mask, tgt_mask,
								src_lengths, tgt_lengths)

	loss = loss_compute(pre_output, tgt_pred, tgt_lengths.sum())

	return loss/tgt_lengths.sum()


def evaluate(model, loader, loss_compute, device, task_id=0):
	'''
		This method performs an evaluation on all dataset
		model: the model being evaluated
		loader: dataloader that provides the batches
		loss_compute: function to compute the loss
		device: if use gpu or cpu
		task_id: task id that is being trained (0 as default)
	'''
    
	model.eval()  
	epoch_loss = 0
	total_tokens = 0
	with torch.no_grad():

		for i, (src, tgt, src_mask, tgt_mask, src_lengths, tgt_lengths, tgt_pred) in enumerate(loader):		

			src = src.to(device)
			tgt = tgt.to(device)
			src_mask = src_mask.to(device)
			tgt_mask = tgt_mask.to(device)
			tgt_pred = tgt_pred.to(device)

			out, _, pre_output = model.forward(src, tgt,
											src_mask, tgt_mask,
											src_lengths, tgt_lengths)

			loss = loss_compute(pre_output, tgt_pred, tgt_lengths.sum())
			epoch_loss += loss
			total_tokens += tgt_lengths.sum()

	return epoch_loss / total_tokens   	


def run_translate(model, source_vocab, target_vocabs, save_dir, device, beam_size, filenames, max_length):
	'''
		This method builds a model from scratch or using the encoder of a pre-trained model
		model: the model being evaluated
		source_vocabs: the source vocabulary for each file
		target_vocabs: the target vocabulary for each file
		save_dir: path where the outpus will be saved
		beam_size: beam size during the translating
		filenames: filenames of triples to process
		max_length: max length of a sentence
	'''


	for index, eval_name in enumerate(filenames):
		n = len(eval_name.split("/"))
		name = eval_name.split("/")[n-1]
		print(f'Reading {eval_name}')
		fout = open(save_dir + name + "." + str(index) + ".out", "w")
		with open(eval_name, "r") as f:
			outputs = translate(model, index, f, source_vocab, target_vocabs[index], device, 
							beam_size=beam_size, max_length=max_length, type="rnn")
			for output in outputs:
				fout.write(output.replace("<eos>","").strip() + "\n")
		fout.close()


def train(args):

	set_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	batch_size = args.batch_size
	max_length = args.max_length
	mtl = args.mtl

	learning_rate = 0.0005
	if not args.learning_rate:
		learning_rate = args.learning_rate

	if len(args.train_source) != len(args.train_target):
		print("Error.Number of inputs in train are not the same")
		return

	if len(args.dev_source) != len(args.dev_target):
		print("Error: Number of inputs in dev are not the same")
		return

	if not args.tie_embeddings:
		print("Building Encoder vocabulary")
		source_vocabs = build_vocab(args.train_source, args.src_vocab, save_dir=args.save_dir)
		print("Building Decoder vocabulary")
		target_vocabs = build_vocab(args.train_target, args.tgt_vocab, mtl=mtl, name ="tgt", save_dir=args.save_dir)
	else:
		print("Building Share vocabulary")
		source_vocabs = build_vocab(args.train_source + args.train_target, args.src_vocab, name="tied", save_dir=args.save_dir)
		if mtl:
			target_vocabs = [source_vocabs[0] for _ in range(len(args.train_target))]
		else:
			target_vocabs = source_vocabs
	print("Number of source vocabularies:", len(source_vocabs))
	print("Number of target vocabularies:", len(target_vocabs))

	save_params(args, args.save_dir + "args.json")


	print("Building training set and dataloaders")
	train_loaders = build_dataset(args.train_source, args.train_target, batch_size, \
			source_vocabs=source_vocabs, target_vocabs=target_vocabs, shuffle=True, mtl=mtl, max_length=max_length)
	for train_loader in train_loaders:
		print(f'Train - {len(train_loader):d} batches with size: {batch_size:d}')

	print("Building dev set and dataloaders")
	dev_loaders = build_dataset(args.dev_source, args.dev_target, batch_size, \
			source_vocabs=source_vocabs, target_vocabs=target_vocabs, mtl=mtl, max_length=max_length)
	for dev_loader in dev_loaders:
		print(f'Dev - {len(dev_loader):d} batches with size: {batch_size:d}')

	if args.model is not None:
		print("Loading the encoder from an external model...")
		seq2seq_model = load_model(args, source_vocabs, target_vocabs, device, max_length)
	else:
		print("Building model")
		seq2seq_model = build_model(args, source_vocabs, target_vocabs, device, max_length)

	print(f'The Sequence-to-Sequence has {count_parameters(seq2seq_model):,} trainable parameters')
	print(f'The Source Embeddings has {count_parameters(seq2seq_model.src_embed):,} trainable parameters')
	print(f'The Target Embeddings has {count_parameters(seq2seq_model.trg_embed):,} trainable parameters')
	print(f'The Generator has {count_parameters(seq2seq_model.generator):,} trainable parameters')
	print(f'The Encoder has {count_parameters(seq2seq_model.encoder):,} trainable parameters')
	print(f'The Decoder has {count_parameters(seq2seq_model.decoder):,} trainable parameters')


	criterions = [LabelSmoothing(size=target_vocab.len(), padding_idx=constants.PAD_IDX, smoothing=0.1) \
                                        for target_vocab in target_vocabs]

	# Default optimizer
	optimizer = torch.optim.Adam(seq2seq_model.parameters(), betas=(0.9, 0.98), eps=1e-09)
	#model_opts = [NoamOpt(args.hidden_size, args.warmup_steps, optimizer) for _ in target_vocabs]
	model_opts = [optimizer for _ in target_vocabs]

	task_id = 0
	print_loss_total = 0  # Reset every print_every

	n_tasks = len(train_loaders)
	best_valid_loss = [float('inf') for _ in range(n_tasks)]

	if not args.translate:
		print("Start training...")
		patience = 30
		if not args.patience:
			patience = args.patience

		for _iter in range(1, args.steps + 1):

			train_loss = train_step(seq2seq_model, train_loaders[task_id], \
                       LossCompute(seq2seq_model.generator, criterions[task_id], model_opts[task_id]), device)

			print_loss_total += train_loss

			if _iter % args.print_every == 0:
				print_loss_avg = print_loss_total / args.print_every
				print_loss_total = 0
				print(f'Task: {task_id:d} | Step: {_iter:d} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}') #}

			if _iter % args.eval_steps == 0:
				print("Evaluating...")
				valid_loss = evaluate(seq2seq_model, dev_loaders[task_id], LossCompute(seq2seq_model.generator, criterions[task_id], None), \
	                            device, task_id=task_id)
				print(f'Task: {task_id:d} | Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

				if valid_loss < best_valid_loss[task_id]:
					print(f'The loss decreased from {best_valid_loss[task_id]:.3f} to {valid_loss:.3f} in the task {task_id}... saving checkpoint')
					patience = 30
					best_valid_loss[task_id] = valid_loss
					torch.save(seq2seq_model.state_dict(), args.save_dir + 'model.pt')
					print("Saved model.pt")
				else:
					if n_tasks == 1:
						if patience == 0:
							break
						else:
							patience -= 1

				if n_tasks > 1:
					print("Changing to the next task ...")
					task_id = (0 if task_id == n_tasks - 1 else task_id + 1)


	try:
		seq2seq_model.load_state_dict(torch.load(args.save_dir + 'model.pt'))
	except:
		print(f'There is no model in the following path {args.save_dir}')
		return

	print("Evaluating and testing")
	run_translate(seq2seq_model, source_vocabs[0], target_vocabs, args.save_dir, device, args.beam_size, args.eval, max_length=max_length)
	run_translate(seq2seq_model, source_vocabs[0], target_vocabs, args.save_dir, device, args.beam_size, args.test, max_length=max_length)


