# coding=utf-8
import pickle
import os
import re
import json
import numpy as np
from collections import Counter

q_path = 'data/ques_list.pkl'
a_path = 'data/ans_list.pkl'
id_path =  'data/_id.pkl'

split_threshold = 12433
glove300_path = os.path.join('/mnt/md0/user_unilight', 'Data', 'glove.840B.300d.txt')
shared_path = os.path.join('data','shared.json')
data_path = os.path.join('data','data.json')
v_feat_path = os.path.join('data', 'MLDS_hw2_data','training_data','feat')

def read_data(data_type, opts):
	return DataProcessor(data_type, opts)

class DataProcessor:
	def __init__(self, data_type, opts):
		self.data_type = data_type
		self.opts = opts
		global data_path, shared_path, split_threshold, v_feat_path
		data_path = data_path
		shared_path = shared_path
		self.data = self.load_file(data_path)
		self.shared = self.load_file(shared_path)

		if self.data_type == 'train':
			self.data = self.data[:split_threshold]
		else:
			self.data = self.data[split_threshold:]

		self.load_video_feat()

		self.num_samples = self.get_data_size()
		print("Loaded {} examples from {}".format(self.num_samples, data_type))
		print("vocab has size", len(self.shared['vocab']))

	def load_video_feat(self):
		from tqdm import tqdm as tqdm
		video_feat = {}
		for sample in tqdm(self.data):
			if not sample['id'] in video_feat:
				video_feat[sample['id']] = np.load(os.path.join(v_feat_path,sample['id']+'.npy'))
		self.video_feat = video_feat

	def load_file(self, path):
		with open(path, 'r') as fh:
			content = json.load(fh)
		return content

	def get_data_size(self):
		return len(self.data)

	def get_training_batch(self, batch_no):
		opts = self.opts
		si = (batch_no * opts['batch_size'])
		ei = min(self.num_samples, si + opts['batch_size'])
		n = ei - si

		tensor_dict = {}
		video = np.zeros((n, opts['v_length'], opts['v_dim']))
		question = np.zeros((n, opts['q_length'], opts['q_dim']))
		answer = np.zeros( (n, opts['vocab']) )
		idxs = []

		count = 0
		for i in range(si, ei):
			idxs.append(i)
			sample = self.data[i]
			q = sample['q']

			video[count] = self.video_feat[sample['id']]

			for j in range(len(q)):
				try:
					word = self.shared['vocab'][q[j]]
					question[count][j] = self.shared['w2v'][word]
				except KeyError:
					pass

			answer[count][sample['a'][0]] = 1.0
			
			count += 1
		
		tensor_dict['video'] = video
		tensor_dict['question'] = question
		tensor_dict['answer'] = answer
		return tensor_dict, idxs
	
	def get_testing_batch(self, batch_no):
		opts = self.opts
		si = (batch_no * opts['batch_size'])
		ei = min(self.num_samples, si + opts['batch_size'])
		n = ei - si

		video = np.zeros((n, opts['v_length'], opts['v_dim']))
		question = np.zeros((n, opts['q_length'], opts['q_dim']))
		answer = np.zeros( (n, opts['vocab']) )
		
		count = 0
		for i in range(si, ei):
			sample = self.data[i]
			q = sample['q']

			video[count] = self.video_feat[sample['id']]

			for j in range(len(q)):
				try:
					word = self.shared['vocab'][q[j]]
					question[count][j] = self.shared['w2v'][word]
				except KeyError:
					pass

			answer[count][sample['a'][0]] = 1.0
			
			count += 1
		
		return video, question, answer, n


def get_word2vec(glove_path, vocab):
	w2v = {}
	w2v_dict = {}
	with open(glove_path, 'r', encoding='utf-8') as fh:
		for line in fh:
			array = line.lstrip().rstrip().split(" ")
			word = array[0]
			vector = list(map(float, array[1:]))
			w2v[word] = vector

	count = 0
	zero_vector = [0.0 for _ in range(300)]
	for idx, word in enumerate(vocab):
		if word in w2v:
			w2v_dict[word] = w2v[word]
			count +=1
		else:
			w2v_dict[word] = zero_vector

	print("{}/{} of word vocab have corresponding vectors in {}".format(count, len(vocab), glove_path))

	return w2v_dict

def build_vocab():
	import nltk
	nltk.download('punkt')
	from nltk.tokenize import word_tokenize

	with open(q_path, 'rb') as f:
		q_list = pickle.load(f)
	with open(a_path, 'rb') as f:
		a_list = pickle.load(f)

	word_counter = Counter()
	for q_video in q_list:
		for q in q_video:
			tokens = word_tokenize(q)
			for w in tokens:
				word_counter[w] += 1
	
	for a_video in a_list:
		for a in a_video:
			tokens = word_tokenize(a)
			for w in tokens:
				word_counter[w] += 1

	vocab = list(word_counter)
	print('Building w2v from Glove...')
	w2v = get_word2vec(glove300_path, vocab)
	shared = {'vocab': vocab,
			  'w2v': w2v,
			  }

	print('Saving...')
	with open(shared_path, 'w') as f:
		json.dump(shared, f)
	
def load_file(path):
	with open(path, 'r') as fh:
		content = json.load(fh)
	return content

def parse():
	import nltk
	nltk.download('punkt')
	from nltk.tokenize import word_tokenize
	
	shared = load_file(shared_path)
	vocab_dict = {w:idx for idx, w in enumerate(shared['vocab'])}
	with open(q_path, 'rb') as f:
		q_list = pickle.load(f)
	with open(a_path, 'rb') as f:
		a_list = pickle.load(f)
	with open(id_path, 'rb') as f:
		id_list = pickle.load(f)
	
	data = []
	max_q = 0
	for idx1, q_video in enumerate(q_list):
		for idx2, q in enumerate(q_video):
			idx_list_q = []
			idx_list_a = []
			tokens_q = word_tokenize(q)
			a = a_list[idx1][idx2]
			tokens_a = word_tokenize(a)
			for w in tokens_q:
				idx_list_q.append(vocab_dict[w])
			for w in tokens_a:
				idx_list_a.append(vocab_dict[a])
			
			max_q = len(tokens_q) if len(tokens_q) > max_q else max_q
			
			sample = {'id': id_list[idx1], 
					  'q': idx_list_q,
					  'a': idx_list_a,
					  }
			data.append(sample)

	print(max_q)
	print('{} samples'.format(len(data)))
	print('Saving...')
	with open(data_path, 'w') as f:
		json.dump(data, f)

def run():
	import sys
	from pprint import pprint
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--build_vocab', type=bool, default=False, help='build vocab and glove from q and a list')
	parser.add_argument('--parse', type=bool, default=False, help='original q and a list to seq')
	parser.add_argument('--debug', type=bool, default=False, help='load set for debug')
	args = parser.parse_args()
	
	if args.build_vocab:
		print('Building Vocab...')
		build_vocab()
	if args.parse:
		print('Parsing...')
		parse()
	


if __name__ == '__main__':
	run()

