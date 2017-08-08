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
	def __init__(self,input_size,output_size,learning_rate=0.5,iterate=100):
		super(MalwareBehaviorRiskScore, self).__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._learning_rate = learning_rate
		self._iterate = iterate

		self.neural_network()
		
	def neural_network(self):
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
		
		
		layer_1 = tf.nn.sigmoid(tf.matmul(self._X ,self._hidden_1_layer['weights'])) + self._hidden_1_layer['biases']
		self._prediction = tf.nn.sigmoid(tf.matmul(layer_1 ,self._hidden_2_layer['weights'])) + self._hidden_2_layer['biases']

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

			for epoch in range(0,self._iterate):
				sess.run(self._optimizer,feed_dict={self._X:X,self._Y_:desires})
				train_accuacy = self._accuracy.eval(feed_dict={self._X: X, self._Y_: desires})
				print("step %d, training accuracy %g"%(epoch, train_accuacy))
			
			

	def inference(X):
		print 'inference'

	def saveModel(self,filename='model'):
		print 'saveModel'

	def restoreModel(self,filename='model'):
		print 'restoreModel'

if __name__ == "__main__":
    #vec_n = PrePorcess.getVector(filename='../samples/Malware_Samples/ASM_Malekal/7f7ccaa16fb15eb1c7399d422f8363e8.asm')
	X = MalwareSamples('../samples/Malware_Samples/ASM_Malekal')

	model = MalwareBehaviorRiskScore(input_size=3,output_size=1,iterate=500)
	
	
	x_list = []
	x_list.append(np.array([0,0,1]))
	x_list.append(np.array([1,0,0]))
	x_list.append(np.array([0,0,1]))
	x_list.append(np.array([1,0,0]))
	x_list.append(np.array([0,1,1]))
	x_list.append(np.array([1,1,1]))
	
	y = []
	y.append(np.array([1]))
	y.append(np.array([1]))
	y.append(np.array([1]))
	y.append(np.array([1]))
	y.append(np.array([1]))
	y.append(np.array([0]))


	model.training(x_list,y)