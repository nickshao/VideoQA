import preprocess
from Models.model_rstan import RSTAN
import numpy as np
import tensorflow as tf
import argparse
import random
import string
import os

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
	parser.add_argument('--epochs', type=int, default=12, help='Expochs')
	parser.add_argument('--debug', type=bool, default=False, help='print debug msgs')
	parser.add_argument('--load', type=bool, default=False, help='load model')
	parser.add_argument('--save_dir', type=str, default='Models/save/', help='Data')

	args = parser.parse_args()

	modOpts = {
		'is_training' : True,
		'in_keep_prob': 0.6,
		'batch_size' : args.batch_size,
		'v_length' : 80, # video length
		'q_length' : 30, # question length
		'vocab': 4220,
		'v_dim': 4096,
		'q_dim': 300,
		'bGRU_dim': 512,
		'aGRU_dim': 512,
	}

	print('Reading data')
	dp = preprocess.read_data('train', modOpts)
	num_batches = int(np.floor(dp.num_samples/args.batch_size)) - 1
	
	rstan_model = RSTAN(modOpts)
	input_tensors, loss, acc, pred = rstan_model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)
	#train_op = tf.train.AdadeltaOptimizer(1.0, rho=0.95, epsilon=1e-06,).minimize(loss)

	#saver
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	saver = tf.train.Saver(max_to_keep = 10)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	if args.load:
		PATH = 'Models/save/rstan_model29.ckpt'
		start_epoch = 30
		saver.restore(sess, PATH)
		f = open('results/rstan_training_result.txt','a')
	else:
		init = tf.global_variables_initializer()
		sess.run(init)
		f = open('results/rstan_training_result.txt','w')
		start_epoch = 0

	for i in range(start_epoch, start_epoch + args.epochs):
		rl=random.sample(range(num_batches), num_batches)
		batch_no = 0
		LOSS = 0.0
		EM = 0.0
		while batch_no < num_batches:
			tensor_dict, idxs = dp.get_training_batch(rl[batch_no])
			feed_dict = {
				input_tensors['v']:tensor_dict['video'],
				input_tensors['q']:tensor_dict['question'],
				input_tensors['a']:tensor_dict['answer'],
			}
			_, loss_value, accuracy, predictions = sess.run(
					[train_op, loss, acc, pred], feed_dict=feed_dict)
			batch_no += 1
			LOSS += loss_value
			EM += accuracy
			print("{} epoch {} batch, Loss:{:.2f}, Acc:{:.2f}".format(i, batch_no, loss_value, accuracy))
		save_path = saver.save(sess, os.path.join(args.save_dir, "rstan_model{}.ckpt".format(i)))
		f.write(' '.join( ("Loss", str(LOSS/num_batches), str(i), '\n' ) ) )
		f.write(' '.join( ("EM", str(EM/num_batches), '\n') ) )
		f.write("---------------\n")
		f.flush()
		print("---------------")
	f.close()
	save_path = saver.save(sess, os.path.join(args.save_dir, "rstan_model_final.ckpt"))
	print('save path:',save_path)


if __name__ == '__main__':
	run()
