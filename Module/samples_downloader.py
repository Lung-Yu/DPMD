import requests
from BeautifulSoup import BeautifulSoup
import logging
import os
BASE_URL = "http://malwaredb.malekal.com/"
URL_MALEKAL = "http://malwaredb.malekal.com/index.php?page="


def downloadFile(file_url,filename):
	shell = 'wget ' + file_url + " -O ./samples/malekal/" + filename+".zip"
	print shell
	os.system(shell)

def getPage(pageIndex):
	#html parser
	print 'download page : ' + str(pageIndex)

	full_url = BASE_URL + str('?page=') + str(pageIndex)
	res = requests.post(full_url)
	soup = BeautifulSoup(res.text)
	tds = soup.body.findAll('td')

	#get all file
	for i in range(0,len(tds),+8):
		file = tds[i].findAll('a')[0]['href']
		filename = file[17:] 
		url = BASE_URL + file[1:]
		downloadFile(url,filename)

def main():
	for i in range(1,61):
		getPage(i)


if __name__ == '__main__':
	main()
