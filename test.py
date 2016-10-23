#encoding = "utf-8"
#!/usr/bin/env python

import re
import math
from sklearn import svm

def test() :
	word = "1sfv2"
	content = "weJusff14D"
	flag = re.match( r'.*(\d)+.*' , content ) 
	if flag :
		print "Yes is good" 

def test1() :
	x1 = 14.0 / 16
	x2 = 1.0 / 16
	x3 = 1.0 / 16
	x = ( - 1 ) * x1 * math.log( x1 ) + ( - 1 ) * x2 * math.log( x2 ) + ( - 1 ) * x3 * math.log( x3 )
	print x

def test2() :
	x = [[1,2],[2,3]]
	y = [1,3]
	model = svm.SVC()
	model.fit( x , y )
	xx = [ 4 ,5]
	print model.predict( xx )

if __name__ == '__main__' :
	test2()