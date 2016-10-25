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


def process() :
	freStopwordsPath = "../configFile/usedFrequentStopwords"   #ͣ�ô����ڵ�·��
	docPath = "../sampleData/gubaTraindata"               #�ĵ����ڵ�·��
	stopwords = codecs.open( freStopwordsPath , 'r' , encoding = 'utf8' ).readlines()   #��ȡͣ�ô��б�
	stopwords = { w.strip().decode( "utf-8" ) for w in stopwords }    #ͣ�ôʼ���

	train_set = list()

	with open( docPath , 'r' ) as f :
		for line in f :
			fields = line.strip().split( "," , 1 )
			if len( fields ) != 2 :
				continue
			content = fields[ 1 ]    #�ĵ��ı�ǩ
			content = list( jieba.cut( content , cut_all = False ) )   #��ͽ��зִ�

			cc = [ w for w in content if w not in stopwords and ( not re.match( r'.*(\w|\d)+.*' , unicodedata.normalize('NFKD', w ).encode('ascii','ignore') ) ) and len( w ) > 1 ]
                                                   #ȥ��ͣ�ôʣ����ֺ��ַ� �����ڳ��Ƚ�С��Ҳȥ��
			train_set.append( cc )


	dictionary = Dictionary( train_set )        #�õ����дʵļ��ϣ��Ǹ��ֵ䣬{�ʣ��ʵ�����}
	corpus = [ dictionary.doc2bow( text ) for text in train_set ]   #�õ��ĵ�����

	lda = LdaModel( corpus = corpus , id2word = dictionary , num_topics = 20 , alpha = 'auto' ) #������������

	# lda.print_topics( 10 )
	output_file = './../processData/lda_output'  
	for pattern in lda.show_topics( 20 , 20 ):   #ע�⣬pattern �Ǹ�Ԫ�棬��ʽ�ǣ����⣨���ͣ� ���ʵĸ���*�� ��unicode��ʽ��show_topics�е�ǰһ������������������һ������������������µĴʸ���
		print pattern[ 1 ].encode( "utf-8" ) 
	# print lda.show_topics()   

if __name__ == '__main__' :
	begin = time.clock()
	process()
	end = time.clock()
	print "The running time is : " , ( end - begin )
