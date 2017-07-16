import binascii
filename = 'cv2.exe'

class PreProcess(object):
	"""docstring for PreProcess"""
	def __init__(self):
		super(PreProcess, self).__init__()
		
		self._hex_code = ''

	@staticmethod
	def hex2wordVec(code):
		i = int(code,16)
		words = []
		while i != 0:
			val = i % 2
			i = i / 2
			words.append(val)
		words = words[::-1]
		while len(words) < 4:
			words.insert(0, 0)
		return words

	def loadfile(self,filename):
		
		with open(filename, 'rb') as f:
			self._hex_code = binascii.hexlify(f.read())

	def process(self):
		wordVec = [[]]
		for i in range(0,len(self._hex_code),2):
			code1 = self.hex2wordVec(self._hex_code[i])
			code2 = self.hex2wordVec(self._hex_code[i+1])
			wordVec.append(code1 + code2)
		return wordVec

def main():
	p = PreProcess()
	p.loadfile(filename)
	wordVec = p.process()
	print wordVec[1:2]

main()
