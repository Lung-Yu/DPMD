import re

class ObjdumpParser(object):
	"""docstring for ObjdumpParser"""
	def __init__(self):
		super(ObjdumpParser, self).__init__()

	def loadfile(self,filename):
		with open(filename, mode='r') as f :
			self._fileContent = f.read()
	
	def getContnet(self):
		tag_start = '00401000 <.text>:'
		index = self._fileContent.find(tag_start)
		self._content = self._fileContent[index + len(tag_start)+1:]
		return self._content

	def getItems(self,content):
		self._items = re.findall(r"(.{6}:.{18})",content)
		return self._items

	def getStepVectors(self,items):

		self._steps = []
		for item in items:
			index = item.find(":")

			word_line = item[index+2:]
			print word_line

			step_vetor = []
			for i in range(0,len(word_line),3):
			 	word = word_line[i:i+2]
			 	print word
			 	vec = self.hex2wordVec(word[0]) + self.hex2wordVec(word[1])
			 	step_vetor = step_vetor + vec

			self._steps.append(step_vetor)
		return self._steps

	def asm2vec(self):
		self.getContnet()
		self.getItems(self._content)
		self.getStepVectors(self._items)
		return self._steps


	@staticmethod
	def hex2wordVec(code):
		if code == ' ':
			return [1,0, 0, 0,0]
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

def main():
	filename = 'asm.txt'
	parser = ObjdumpParser()
	parser.loadfile(filename)
	for step in parser.asm2vec():
			print step

if __name__ == '__main__':
	main()
