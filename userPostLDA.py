#encoding = "utf-8"
#!/usr/bin/env python

import sys
import time
from LDAdemo import process , printModel

reload( sys )
sys.setdefaultencoding( "utf-8" )

def testDemo( dataSetPath ) :
	lda = process( dataSetPath , 60 )
	printModel( lda , 60 , 30 )

if __name__ == '__main__' :
	begin = time.clock()

	dataSetPath = "../totalData/allData"
	testDemo( dataSetPath ) 

	end = time.clock()
	print "The total cost of time is : %f"%( end - begin )