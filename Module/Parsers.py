import re
import sys
import os
import numpy as np

class FormateNotMatch(Exception):
	"""docstring for FormateNotMatch"""
	pass


class ObjdumpParser(object):
	"""docstring for ObjdumpParser"""
	def __init__(self):
		super(ObjdumpParser, self).__init__()

	def loadfile(self,filename):
		with open(filename, mode='r') as f :
			#self._fileContent = f.read()
			self._contentLines = f.readlines()

	def getContnet(self):
		#tag_start = '00401000 <.text>:'
		tag_start = "<.text>:"
		index = self._fileContent.find(tag_start)
		self._content = self._fileContent[index + len(tag_start)+1:]
		#print self._content
		return self._content

	def getItems(self,content):
		self._items = re.findall(r"(.{6}:.{18})",content)
		return self._items

	def getStepVectors(self,items):
		self._steps = []
		for item in items:
			index = item.find(":")

			word_line = item[index+2:]
			#print word_line
			#print item

			step_vetor = []
			try:
				for i in range(0,len(word_line),3):
				 	word = word_line[i:i+2]
				 	#print word
				 	vec = self.hex2wordVec(word[0]) + self.hex2wordVec(word[1])
				 	step_vetor = step_vetor + vec
				self._steps.append(np.array(step_vetor))
			except FormateNotMatch as e :
				print e.message
				del e
		return self._steps

	def asm2vec(self):
		#self.getContnet()
		#self.getItems(self._content)
		#self.getStepVectors(self._items)
		self._steps = []

		#get word vector by lines
		isStart = False
		for line in self._contentLines:
			#pre-process
			if line.find("<.text>:") > 0:
				isStart = True

			if not isStart :
				continue

			index = line.find(":") + 2
			word_line = line[index:index+18]
			del line,index
			if len(word_line) < 18:
				del word_line
				continue

			#convert word to vector by one-hot-encoding
			step_vetor = []
			for i in range(0,len(word_line),3):
				word = word_line[i:i+2]
				vec = self.hex2wordVec(word[0]) + self.hex2wordVec(word[1])
				#print ("len:%d ,index=%02d,wordline=%s , word=%s -> [%s]")%(len(word_line),i,str(word_line),str(word),("".join([str(e) for e in vec])))
				step_vetor = step_vetor + vec
				#print word_line + "->" + "".join(str(e) for e in vec)
			self._steps.append(np.array(step_vetor))

			#print ("".join([str(e) for e in np.array(step_vetor)]))
			#print np.array(step_vetor)
		#print self._steps
		# for s in self._steps:
		# 	print len(s)
		return self._steps
		#return self._steps


	@staticmethod
	def hex2wordVec(code):
		if code == ' ':
			return [1 ,0 ,0 ,0 ,0]

		i = int(code,16)
		words = []
		while i != 0:
			val = i % 2
			i = i / 2
			words.append(val)
		words = words[::-1]
		while len(words) < 5:
			words.insert(0, 0)

		return words

#FEATURE_DIR = './features'
def main(filename='asm.txt'):
	#filename = 'asm.txt'
	parser = ObjdumpParser()
	parser.loadfile(filename)

	# with open(new_filename,'w') as f:
	# 	for step in parser.asm2vec():
	# 			f.write(str(step))
	for item in parser.asm2vec():
		#print item
		#print ("len=%d -> [%s]")%(len(item),("".join([str(e) for e in item])))
		pass
if __name__ == '__main__':
	#main(sys.argv[1])
	main('../Malware_Samples/ASM_Malekal/66d71817551be082f0b2e1ea7af444fc.asm')


