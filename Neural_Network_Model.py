import tensorflow as tf
import numpy as np
from Modules.Parsers import ObjdumpParser
from Modules.Parsers import PeParser
import os

class PrePorcess(object):
	"""docstring for PrePorcess"""
	def __init__(self):
    		super(PrePorcess, self).__init__()		

	@staticmethod
	def getVector(filename):
		parser = ObjdumpParser()
		parser.loadfile(filename)
		asm_vec = parser.asm2vec()
		#print type(asm_vec)
		vector = np.array(asm_vec)
		#print type(vector)
		del asm_vec,parser
		return vector

class MalwareSamples(object):
    	"""docstring for PrePorcess"""
	def __init__(self,DATA_DIR):
		super(MalwareSamples, self).__init__()
		self._DIR_DIR = DATA_DIR

	def getVector(self,filename):
		parser = ObjdumpParser()
		parser.loadfile(filename)
		asm_vec = parser.asm2vec()
		#print type(asm_vec)
		vector = np.array(asm_vec)
		#print type(vector)
		del asm_vec,parser
		return vector

	def nextBatch(self,batch_size=1):
		files = np.random.choice(os.listdir(self._DIR_DIR),replace=False,size=batch_size)
		res = []

		for i in xrange(0,batch_size):
			vector = self.getVector(os.path.join(self._DIR_DIR,files[i]))
			if len(vector) != 0:
				res.append(vector)
				
		return res


class MalwareBehaviorRiskScore(object):
    	"""docstring for PrePorcess"""
	def __init__(self,input_size,output_size,learning_rate=0.5,iters=100):
		super(MalwareBehaviorRiskScore, self).__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._learning_rate = learning_rate
		self._iters = iters

		self.neural_network()
		
	def neural_network(self,rnn_cell_size=10):
		self._hidden_1_layer = {
			'weights':tf.Variable(tf.random_uniform( [self._input_size,128],-1.0 ,1.0 ),name='weights'),
			'biases':tf.Variable(tf.zeros([1]) ,name='biases')
		}
		
		self._hidden_2_layer = {
			'weights':tf.Variable(tf.random_uniform( [128,self._output_size],-1.0 ,1.0 ),name='weights'),
			'biases':tf.Variable(tf.zeros([1]) ,name='biases')
		}

		with tf.name_scope('input'):
			self._X = tf.placeholder(tf.float32, [None, self._input_size], name='X')
			self._Y_ = tf.placeholder(tf.float32, [None, self._output_size], name='Y_')
		
		self._step = tf.placeholder(tf.float32,rnn_cell_size)
		
		
		layer_1 = tf.nn.sigmoid(tf.matmul(self._X ,self._hidden_1_layer['weights']) + self._hidden_1_layer['biases'])
		self._prediction =tf.nn.sigmoid( tf.matmul(layer_1, self._hidden_2_layer['weights']) + self._hidden_2_layer['biases'])
		
		self.lost_func()
		self.optimizer()
   	

	def optimizer(self):
		self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._loss_func)  #learning rate

	def lost_func(self):
		#self._loss_func = tf.reduce_mean(tf.reduce_sum(tf.square(self._Y_ - self._prediction)))
		self._loss_func = tf.reduce_mean(tf.square(self._Y_ - self._prediction))
		self._accuracy = tf.reduce_mean(tf.cast(self._prediction, tf.float32))
	
	def training(self,X,desires):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(0,self._iters):
				sess.run(self._optimizer,feed_dict={self._X:X,self._Y_:desires})
				train_accuacy = self._accuracy.eval(feed_dict={self._X: X, self._Y_: desires})
				print("step %d, training accuracy %g"%(epoch, train_accuacy))

			print ("finish.")
			
	def inference(X):
		print 'inference'

	def saveModel(self,sess,filename='/tmp/model.ckpt'):
		print 'saveModel'

	def restoreModel(self,filename='/tmp/model.ckpt'):
		print 'restoreModel'





class MalwareBehaviorFeature(object):
    	"""docstring for PrePorcess"""
	def __init__(self,input_size,output_size,learning_rate=0.5,iters=100):
		super(MalwareBehaviorFeature, self).__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._learning_rate = learning_rate
		self._iters = iters
		
	def neural_network(self,rnn_cell_size=10):
		self._hidden_1_layer = {
			'weights':tf.Variable(tf.random_uniform( [self._input_size,512],-1.0 ,1.0 ),name='weights'),
			'biases':tf.Variable(tf.zeros([1]) ,name='biases')
		}
		
		self._hidden_2_layer = {
			'weights':tf.Variable(tf.random_uniform( [512,128],-1.0 ,1.0 ),name='weights'),
			'biases':tf.Variable(tf.zeros([1]) ,name='biases')
		}

		self._hidden_3_layer = {
			'weights':tf.Variable(tf.random_uniform( [128,64],-1.0 ,1.0 ),name='weights'),
			'biases':tf.Variable(tf.zeros([1]) ,name='biases')
		}

		self._hidden_4_layer = {
			'weights':tf.Variable(tf.random_uniform( [64,self._output_size],-1.0 ,1.0 ),name='weights'),
			'biases':tf.Variable(tf.zeros([1]) ,name='biases')
		}

		with tf.name_scope('input'):
			self._X = tf.placeholder(tf.float32, [None, self._input_size], name='X')
			self._Y_ = tf.placeholder(tf.float32, [None, self._output_size], name='Y_')
		
		
		layer_1 = tf.nn.relu(tf.matmul(self._X ,self._hidden_1_layer['weights']) + self._hidden_1_layer['biases'])
		# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(512,forget_bias=1.0,state_is_tuple=True,activation=tf.nn.tanh)
		# init_state = lstm_cell.zero_state(self._rnn_batch, dtype=tf.float32) # init all value to zero state
		# layer_rnn, final_state = tf.nn.dynamic_rnn(lstm_cell, layer_1, initial_state=init_state, time_major=True)
		layer_2 = tf.nn.relu( tf.matmul(layer_1, self._hidden_2_layer['weights']) + self._hidden_2_layer['biases'])
		layer_3 = tf.nn.relu( tf.matmul(layer_2, self._hidden_3_layer['weights']) + self._hidden_3_layer['biases'])
		self._prediction = tf.nn.sigmoid( tf.matmul(layer_3, self._hidden_4_layer['weights']) + self._hidden_4_layer['biases'])
		# layer_1 = tf.nn.sigmoid(tf.matmul(self._X ,self._hidden_1_layer['weights']) + self._hidden_1_layer['biases'])
		# layer_2 = tf.nn.sigmoid(tf.matmul(self._X ,self._hidden_2_layer['weights']) + self._hidden_2_layer['biases'])
		# self._prediction =tf.nn.sigmoid( tf.matmul(layer_2, self._hidden_3_layer['weights']) + self._hidden_3_layer['biases'])
		
		# self.lost_func()
		# self.optimizer()
		self.define_optimizer()
		self.evaluate_model()
		

	def define_optimizer(self):
		self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._Y_))
		#self._cost = tf.reduce_prod(self._prediction - self._Y_)
		self._train_step = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(self._cost)
		
		
	def evaluate_model(self):
		correct_pred = tf.equal(tf.argmax(self._Y_,1), tf.argmax(self._prediction,1))
		#correct_pred = tf.reduce_prod(self._prediction - self._Y_ )
		self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	def training(self,data_Obj):
		pre_batch,cur_batch = self.getBetch(data_Obj)
		self._rnn_batch = len(pre_batch)
		self.neural_network()
		self._saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			print "start training..."
			for epoch in range(0,self._iters):
				sess.run(self._train_step,feed_dict={self._X:cur_batch,self._Y_:pre_batch})

				train_accuacy = self._accuracy.eval(feed_dict={self._X:cur_batch,self._Y_:pre_batch})
				if epoch % 100 == 0:
    				# Calculate batch accuracy
					acc = sess.run(self._accuracy, feed_dict={self._X: cur_batch, self._Y_: pre_batch})
					# Calculate batch loss
					loss = sess.run(self._cost,  feed_dict={self._X: cur_batch, self._Y_: pre_batch})

					print("Iter " + str(epoch) + ", Minibatch Loss= {:.6f}".format(loss) ) + ", Training Accuracy= {:.5f}".format(acc)

			print("Optimization Finished!")
			self.saveModel(sess)

	def getBetch(self,data_obj):
		data = data_obj.nextBatch()
		pre_batch = []
		cur_batch = []
		for i in range(1,len(data[0])):
			pre_batch.append(data[0][i - 1])
			cur_batch.append(data[0][i])
		return pre_batch,cur_batch
			
	def inference(X):
		print 'inference'

	def saveModel(self,sess,model_name="./tmp/model.ckpt"):
		save_path = self._saver.save(sess, model_name)
		print "Model saved in file: ", save_path

	def restoreModel(self,model_name="./tmp/model.ckpt"):
		self._saver.restore(sess,model_name)
		print "Model restored."

def getTestData():
	x_list = []
	x_list.append(np.array([0,0,1]))
	# x_list.append(np.array([1,0,0]))
	# x_list.append(np.array([0,0,1]))
	# x_list.append(np.array([1,0,0]))
	# x_list.append(np.array([0,1,1]))
	# x_list.append(np.array([1,1,1]))

	y = []
	y.append(np.array([1]))
	# y.append(np.array([1]))
	# y.append(np.array([1]))
	# y.append(np.array([1]))
	# y.append(np.array([1]))
	# y.append(np.array([0]))

	#X.nextBatch(100)
	#data = X.nextBatch()
	return x_list,y

if __name__ == "__main__":
    #vec_n = PrePorcess.getVector(filename='../samples/Malware_Samples/ASM_Malekal/7f7ccaa16fb15eb1c7399d422f8363e8.asm')
	X = MalwareSamples('../samples/postive/')

	model = MalwareBehaviorFeature(input_size=60,output_size=60,iters=10000)
	model.training(X)