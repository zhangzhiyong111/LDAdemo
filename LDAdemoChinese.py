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
	freStopwordsPath = "../configFile/usedFrequentStopwords"   #停用词所在的路径
	docPath = "../sampleData/gubaTraindata"               #文档所在的路径
	stopwords = codecs.open( freStopwordsPath , 'r' , encoding = 'utf8' ).readlines()   #获取停用词列表
	stopwords = { w.strip().decode( "utf-8" ) for w in stopwords }    #停用词集合

	train_set = list()

	with open( docPath , 'r' ) as f :
		for line in f :
			fields = line.strip().split( "," , 1 )
			if len( fields ) != 2 :
				continue
			content = fields[ 1 ]    #文档的标签
			content = list( jieba.cut( content , cut_all = False ) )   #结巴进行分词

			cc = [ w for w in content if w not in stopwords and ( not re.match( r'.*(\w|\d)+.*' , unicodedata.normalize('NFKD', w ).encode('ascii','ignore') ) ) and len( w ) > 1 ]
                                                   #去除停用词，数字和字符 ，对于长度较小的也去除
			train_set.append( cc )


	dictionary = Dictionary( train_set )        #得到所有词的集合，是个字典，{词：词的序列}
	corpus = [ dictionary.doc2bow( text ) for text in train_set ]   #得到文档向量

	lda = LdaModel( corpus = corpus , id2word = dictionary , num_topics = 20 , alpha = 'auto' ) #各个参数设置

	# lda.print_topics( 10 )
	output_file = './../processData/lda_output'  
	for pattern in lda.show_topics( 20 , 20 ):   #注意，pattern 是个元祖，形式是（主题（整型） ，词的概率*词 ）unicode形式，show_topics中的前一个参数是主题数，后一个参数是输出该主题下的词个数
		print pattern[ 1 ].encode( "utf-8" ) 
	# print lda.show_topics()   

if __name__ == '__main__' :
	begin = time.clock()
	process()
	end = time.clock()
	print "The running time is : " , ( end - begin )
