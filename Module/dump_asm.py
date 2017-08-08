import os

def getasm(full_filename):
	asm = os.popen('objdump -d ' + full_filename).read()
	write_asm(full_filename,asm)

def write_asm(filename,asm_code):
	new_filename = filename + "_asm.txt"
	print new_filename
	with open(new_filename,'w') as f:
		f.write(asm_code)

def getFilenames(DATA_DIR = '../Malware_Samples/Malware_Knowledge_Base'):
	filenames = []
	for filename in os.listdir(DATA_DIR):
	    #print "Loading: %s" % filename
	    #loadFile = open(os.path.join(DATA_DIR, filename), 'rb')
	    abs_filename = os.path.join(DATA_DIR, filename)
	    filenames.append(abs_filename)
	    #loadFile.close()
	return filenames

def main():
	filenames = getFilenames()
	for name in filenames:
		getasm(name)
		pass

if __name__ == '__main__':
	main()