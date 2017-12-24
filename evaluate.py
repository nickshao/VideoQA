import preprocess
from Models.model_rstan import RSTAN
import numpy as np
import tensorflow as tf
import argparse
import random
import string
import json
from pprint import pprint

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', type=str, default='_final', help='Model epoch')
	parser.add_argument('--debug', type=bool, default=False, help='print debug msgs')
	parser.add_argument('--dataset', type=str, default='dev', help='dataset')


	args = parser.parse_args()
	
	modOpts = {
		'is_training' : False,
		'in_keep_prob': 1.0,
		'batch_size' : 347,
		'v_length' : 80, # video length
		'q_length' : 30, # question length
		'vocab': 4220,
		'v_dim': 4096,
		'q_dim': 300,
		'bGRU_dim': 512,
		'aGRU_dim': 512,
	}
	pprint(modOpts)

	print('Reading data')
	if args.dataset == 'train':
		raise NotImplementedError
	elif args.dataset == 'dev':
		dp = preprocess.read_data(args.dataset, modOpts)
    
	model = RSTAN(modOpts)
	input_tensors, loss, acc, pred = model.build_model()
	saved_model = 'Models/save/rstan_model'+args.epoch+'.ckpt'

	num_batches = int(np.ceil(dp.num_samples/modOpts['batch_size']))
	print(num_batches, 'batches')
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	new_saver = tf.train.Saver()
	sess = tf.InteractiveSession(config=config)
	new_saver.restore(sess, saved_model)
	
	pred_data = {}

	ACC = 0.0
	empty_answer_idx = np.ndarray((modOpts['batch_size'], modOpts['vocab']))
	for batch_no in range(num_batches):
		video, question, answer,  n = dp.get_testing_batch(batch_no)
		feed_dict={
			input_tensors['v']:video,
			input_tensors['q']:question,
			input_tensors['a']:answer,
		}
		predictions, accuracy = sess.run([pred, acc], feed_dict=feed_dict)

		answer = np.argmax(answer, axis=1)

		ACC += accuracy
		
		for i in range(n):
			print('Truth:', dp.shared['vocab'][answer[i]], 'Pred:', dp.shared['vocab'][predictions[i]])

		print(batch_no, 'ACC', '{:.5f}'.format(ACC/(batch_no+1)))
	print("---------------")
	print("ACC", ACC/num_batches )
	'''
	with open('results/'+args.model+'_prediction.txt', 'w') as outfile:
	    json.dump(pred_data, outfile, indent=4)
    '''


if __name__ == '__main__':
	run()
