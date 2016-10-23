#encoding="utf-8"
#!/usr/bin/env python

import sys
import jieba
import re
import time
import unicodedata
from preProcess import *
from sklearn.svm import LinearSVC 

reload( sys )
sys.setdefaultencoding( "utf-8" )

def getImportWords( importWordPath ) :
	wordImport = dict()
	i = 0

	with open( importWordPath , 'r' ) as f :
		for line in f :
			wordImport[ line.strip().decode( "utf-8" ) ] = i
			i += 1

	return wordImport

def getWordMI( wordMIPath ) :
	wordMIInfor = dict()

	with open( wordMIPath , 'r' ) as f :
		for line in f :
			fields = line.strip().split( "\t" )
			if len( fields ) != 4 :
				continue
			postiveScore = float( fields[ 1 ] )
			neuralScore = float( fields[ 2 ] )
			negativeScore = float( fields[ 3 ] )

			MIScore = (  postiveScore - neuralScore , postiveScore - negativeScore , negativeScore - neuralScore  )  # we use the sub Value to calculate the mutual information 
			wordMIInfor[ fields[ 0 ].decode( "utf-8" ) ] = MIScore

	return wordMIInfor

def getPuncAndMoodWords( puncAndMoodWordsPath ) :
	puncAndMoodWords = dict()

	i = 0
	with open( puncAndMoodWordsPath , 'r' ) as f :
		for line in f :
			puncAndMoodWords[ line.strip().decode( "utf-8" ) ] = i
			i += 1
	return puncAndMoodWords

def importWordFeature( segListFiter , wordImport ) :
	importWordLen = len( wordImport )
	feature = [ 0 ] * importWordLen

	for word in segListFiter :
		if wordImport.has_key( word ) :
			feature[ wordImport[ word ] ] = 1

	return feature

def wordMIFeature( segListFiter , wordMIInfor ) :
	wordLen = 0
	maxWordMIInfor = [ 0 ] * 3
	maxWordMI = - 1000.0
	totalWordMIInfor = [ 0 ] * 3

	for word in segListFiter :
		if wordMIInfor.has_key( word ) :
			wordLen += 1
			MIScore = wordMIInfor[ word ]
			for i in range( 3 ) :
				totalWordMIInfor[ i ] += MIScore[ i ]
				if maxWordMI < abs( MIScore[ i ] ) :
					maxWordMIInfor = MIScore
					maxWordMI = abs( MIScore[ i ] )

	for i in range( 3 ) :
		totalWordMIInfor[ i ] /= wordLen
	totalWordMIInfor.extend( maxWordMIInfor )

	return totalWordMIInfor

def specialWordFeature( segListFiter , puncAndMoodWords ) :
	feature3 = [ 0 ] * len( puncAndMoodWords )

	for word in segListFiter :
		if puncAndMoodWords.has_key( word ) :
			feature3[ puncAndMoodWords[ word ] ] = 1

	return feature3

def sentenceToFeature( content , wordImport , wordMIInfor , stopWords , puncAndMoodWords ) :
	seg_list = jieba.cut( content , cut_all = False )
	segListFiter = { word for word in seg_list if word not in stopWords or ( not re.match( r'.*(\w)+.*' , unicodedata.normalize('NFKD', word ).encode('ascii','ignore') ) ) }  #fiter the stopWords

	X_Feature = list()

	feature1 = importWordFeature( segListFiter , wordImport )  # we use the information Gain to select the import words
	feature2 = wordMIFeature( segListFiter , wordMIInfor )       # we use the MI information to weight the relation between the label and words
	# feature3 = specialWordFeature( segListFiter , puncAndMoodWords )  # we will only consider the punction and mood words , this feture is not good , we delete it instead

	X_Feature.extend( feature1 )
	X_Feature.extend( feature2 )
	# X_Feature.extend( feature3 )

	return X_Feature

def getSVMTrainFormat( trainFilePath , wordImport , wordMIInfor , stopWords , puncAndMoodWords ) :
	X = list()
	Y = list()

	with open( trainFilePath , 'r' ) as f :
		for line in f :
			fields = line.strip().split( "," , 1 )
			label = float( fields[ 0 ] )
			content = fields[ 1 ]
			if label == 2 :
				continue

			X_Feature = sentenceToFeature( content , wordImport , wordMIInfor , stopWords , puncAndMoodWords )

			X.append( X_Feature )
			Y.append( label )

	return X , Y

def trainToGetModel( X , Y ) :             # train the sample to get the model 
	model = LinearSVC()  
	model.fit( X , Y )                        # training the svc model
 
	return model

def testDataResult( testFilePath , wordImport , wordMIInfor , stopWords , model , puncAndMoodWords ) :
	totalNum = 0
	sameLabel = 0

	with open( testFilePath , 'r' ) as f :
		for line in f :
			fields = line.strip().split( "," , 1 )
			if len( fields ) != 2 :
				continue
			label = float( fields[ 0 ] )
			content = fields[ 1 ]
			if label == 2 :
				continue

			X_Feature = sentenceToFeature( content , wordImport , wordMIInfor , stopWords , puncAndMoodWords )
			predictLabel = model.predict( X_Feature )

			PreLabel = float( predictLabel[ 0 ] )
			totalNum += 1
			if PreLabel == label :
				sameLabel += 1
			# if label != PreLabel :
			# 	print label , PreLabel , content
	print "totalNum = %d"%totalNum
	print "The right rate is : %f"%( float( sameLabel ) / totalNum )

"""
# the input the file Path
# process is the main program
# X , Y means the train set and test set respectively
"""

def process() :    
	start = time.clock()

	trainFilePath = "../sampleData/gubaTraindata"
	importWordPath = "../processData/importWords"
	wordMIPath = "../processData/MIwords"
	testFilePath = "../sampleData/gubaTestData"
	puncAndMoodWordsPath = "../processData/puncAndMoodWords"

	stopWords = getStopwords()
	wordImport = getImportWords( importWordPath )
	wordMIInfor = getWordMI( wordMIPath )
	puncAndMoodWords = getPuncAndMoodWords( puncAndMoodWordsPath )

	X , Y = getSVMTrainFormat( trainFilePath , wordImport , wordMIInfor , stopWords , puncAndMoodWords )
	model = trainToGetModel( X , Y )        # we train the file to get trainToGetModel

	testDataResult( testFilePath , wordImport , wordMIInfor , stopWords , model , puncAndMoodWords )   # we test our model in testdata to evaluate performance

	end = time.clock()

	print "The total cost of the time is : %f" % ( end - start )

if __name__ == '__main__' :
	process()
