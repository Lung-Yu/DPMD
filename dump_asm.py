import os

filename = "00000fe7b4ba5f4e15b42350e95ca5e0"

def getasm(filename):
	asm = os.popen('objdump -d ' + filename).read()
	print asm


getasm(filename)

