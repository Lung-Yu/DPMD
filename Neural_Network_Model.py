import tensorflow as tf
import numpy as np
from Module.Parsers import ObjdumpParser

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

if __name__ == "__main__":
    vec_n = PrePorcess.getVector(filename='../samples/Malware_Samples/ASM_Malekal/7f7ccaa16fb15eb1c7399d422f8363e8.asm')
    print vec_n