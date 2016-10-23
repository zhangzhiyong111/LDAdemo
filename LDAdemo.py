#encoding="utf-8"
#!/usr/bin/env python

import sys
import os
import codecs
import unicodedata
import jieba
import re
import time
from gensim.corpora import Dictionary
from gensim.models import LdaModel

reload( sys )
sys.setdefaultencoding( "utf-8" )


def process( docPath , numTopic ) :
	freStopwordsPath = "../configFile/usedFrequentStopwords"
	stopwords = codecs.open( freStopwordsPath , 'r' , encoding = 'utf8' ).readlines()
	stopwords = { w.strip().decode( "utf-8" ) for w in stopwords }

	train_set = list()

	with open( docPath , 'r' ) as f :
		for line in f :
			fields = line.strip()
			content = list( jieba.cut( fields , cut_all = False ) )

			cc = [ w for w in content if w not in stopwords and ( not re.match( r'.*(\w|\d)+.*' , unicodedata.normalize('NFKD', w ).encode('ascii','ignore') ) ) and len( w ) > 1 ]

			train_set.append( cc )


	dictionary = Dictionary( train_set )
	corpus = [ dictionary.doc2bow( text ) for text in train_set ]

	lda = LdaModel( corpus = corpus , id2word = dictionary , num_topics = numTopic , alpha = 'auto' )
	
	return lda

def printModel( lda , topicNum , wordNum ) :
	# lda.print_topics( 10 )
	output_file = './../processData/lda_output'  
	for pattern in lda.show_topics( topicNum , wordNum ):
		print pattern[ 1 ].encode( "utf-8" ) 
	# print lda.show_topics()   

if __name__ == '__main__' :
	begin = time.clock()
	docPath = "../sampleData/gubaTraindata"
	lda = process( docPath , 20 )
	printModel( lda , 20 ,20 )
	end = time.clock()
	print "The running time is : " , ( end - begin )