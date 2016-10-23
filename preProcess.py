#encoding="utf-8"
#!/usr/bin/env python

import jieba
import re
import math
import time
import unicodedata
import sys

reload( sys )
sys.setdefaultencoding( 'utf-8' )

def getStopwords() :
	stopWords = set()
	with open( '../configFile/stopWords.txt' , 'r') as f :
		for line in f :
			stopWords.add( line.strip().decode( 'utf-8' ) )
	return stopWords

def getWordDistributed( stopWords , trainDataPath ) :
	i = 0  # control the test case

	postiveWord = dict()
	neuralWord = dict()
	negativeWord = dict()

	with open( trainDataPath ,'r' ) as f :
		for line in f:
			fields = line.strip().split( "," , 1 )
			label = float( fields[ 0 ] )
			content =fields[ 1 ]
			if label == 2 :  # the label of 2 which means we can't give the right label , so we ignore the content
				continue

			seg_list = jieba.cut( content , cut_all = False )
			segListFiter = { word for word in seg_list if word not in stopWords or re.match( r'.*(\w)+.*' , word ) or ( not re.match( r'.*(\w|\d)+.*' , unicodedata.normalize('NFKD', word ).encode('ascii','ignore') ) ) }  #fiter the stopWords

			if label == 1 :
				for word in segListFiter :
					if len( word ) > 10 :
						continue
					postiveWord[ word ] = postiveWord.get( word , 0 ) + 1
			elif label == 0 :
				for word in segListFiter :
					if len( word ) > 10 :
						continue
					neuralWord[ word ] = neuralWord.get( word , 0 ) + 1
			elif label == -1 :
				for word in segListFiter :
					if len( word ) > 10 :
						continue
					negativeWord[ word ] = negativeWord.get( word , 0 ) + 1

	return postiveWord , neuralWord , negativeWord

def CalculateIG( postiveNum , neuralNum , negativeNum , totalNum ) : # calculate the information Gain between 3 labels 
	postiveFloat = float( postiveNum ) / totalNum
	neuralFloat = float( neuralNum ) / totalNum
	negativeFloat = float( negativeNum ) / totalNum

	if postiveFloat != 0 :
		postiveValue = ( - 1) * math.log( postiveFloat ) * postiveFloat 
	if neuralFloat != 0 :
		neuralValue = ( - 1) * math.log( neuralFloat ) * neuralFloat
	if negativeFloat != 0 :
		negativeValue = ( - 1) * math.log( negativeFloat ) * negativeFloat    # we don't user the calculation of log()/log(2), we use the e instead
	return negativeValue + neuralValue + postiveValue

def CalculateMI( postiveNum , neuralNum , negativeNum , totalNum , postiveLen , neuralLen , negativeLen ) : # calculate the MI
	postMIScore = 10000 * float( postiveNum ) / ( totalNum * postiveLen )
	neuralMIScore = 10000 * float( neuralNum ) / ( totalNum * neuralLen )
	negaMIScore = 10000 * float( negativeNum ) / ( totalNum * negativeLen )
	return postMIScore , neuralMIScore , negaMIScore

def process( postiveWord , neuralWord , negativeWord ) :
	
	wordInforGain = dict()            # get the information gain of the words
	wordMI = dict()                   # get the result of the mutual information
	Vocabulay = set( postiveWord.keys() ) & set( neuralWord.keys() ) & set( negativeWord.keys() )

	postiveLen = len( postiveWord )   # calculate length of each dict 
	neuralLen = len( neuralWord )
	negativeLen = len( negativeWord )

	for word in Vocabulay :
		postiveNum = postiveWord.get( word , 0 )
		neuralNum = neuralWord.get( word , 0 )
		negativeNum = negativeWord.get( word , 0 )

		totalNum = postiveNum + neuralNum + negativeNum 
		if totalNum < 6 or re.match( r'.*(\w)+.*' , unicodedata.normalize('NFKD', word ).encode('ascii','ignore') ) :
			continue

		wordIG = CalculateIG( postiveNum , neuralNum , negativeNum , totalNum )
		postMIScore , neuralMIScore , negaMIScore = CalculateMI( postiveNum , neuralNum , negativeNum , totalNum , postiveLen , neuralLen , negativeLen )

		wordMI[ word ] = ( postMIScore , neuralMIScore , negaMIScore )
		wordInforGain[ word ] = wordIG

	return wordInforGain , wordMI
	

def WriteImportWord( wordInforGain , importWordsPath ) :
	sortedWordInfor = sorted( wordInforGain.items(), key = lambda x:x[1] )
	i = 0 
	fr = open( importWordsPath , 'a')
	for ( word , IG ) in sortedWordInfor :
		fr.write( word.encode( "utf-8" ) + "\n" )
		i += 1
		if i >= 600 :
			break
	fr.close() 

def WriteWordsMI( wordMI , wordMIPath ) :
	fr = open( wordMIPath , 'a' )
	for word , MIScore in wordMI.items() :
		fr.write( "\t".join( [ word.encode( "utf-8") , str( MIScore[ 0 ] ) , str( MIScore[ 1 ] ) , str( MIScore[ 2 ] ) , "\n" ] ) )
	fr.close()


def mainProcess() :

	start = time.clock()

	importWordsPath = "../processData/importWords"
	wordMIPath = "../processData/MIwords"
	trainDataPath = "../sampleData/gubaTraindata"

	stopWords = getStopwords()      #get the stopwords
	postiveWord , neuralWord , negativeWord = getWordDistributed( stopWords , trainDataPath )   #we mainly get the putumal information and information gain
	# printInfor( postiveWord , neuralWord , negativeWord )

	wordInforGain , wordMI = process( postiveWord , neuralWord , negativeWord )  # get the information gain and mutual information of the words
	# printInformation( wordInforGain , wordMI , postiveWord , neuralWord , negativeWord )

	WriteImportWord( wordInforGain , importWordsPath )  #write the import words to file
	WriteWordsMI( wordMI , wordMIPath )         #write the MI of words to file

	end = time.clock()
	print "the total time cost is : %f"%( end - start )

if __name__ == '__main__' :
	mainProcess()


