import os

filename = "00000fe7b4ba5f4e15b42350e95ca5e0"

def getasm(filename):
	asm = os.popen('objdump -d ' + filename).read()
	print asm

#getasm(filename)

DATA_DIR = './samples'
def getFilenames():
	filenames = []
	for filename in os.listdir(DATA_DIR):
	    #print "Loading: %s" % filename
	    #loadFile = open(os.path.join(DATA_DIR, filename), 'rb')
	    filenames.append(filename)
	    #loadFile.close()
	return filenames

def main():
	filenames = getFilenames()
	for name in filenames:
		getasm(name)
		pass

if __name__ == '__main__':
	main()