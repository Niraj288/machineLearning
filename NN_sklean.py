from sklearn.neural_network import MLPRegressor
import os
import numpy as np
import math
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xlwt

def and_band():
	# create Trainig Dataset
	train_x=[[x,math.sin(x)] for x in  range(20000)]
	train_y=[math.sin(x) for x in range(20000)]
	 
	#create neural net regressor
	reg=LinearRegression()
	#reg = MLPRegressor(hidden_layer_sizes=(50,),solver="lbfgs")
	reg.fit(train_x,train_y)
	 
	#test prediction
	test_x=[[x,math.sin(x)] for x in  range(201,220,2)]
	test_y=[math.sin(x) for x in range(201,220,2)]
	predict=reg.predict(test_x)
	print "_Input_\t_output_"
	for i in range(len(test_x)):
	    print "  ",test_y[i],"---->",predict[i]

#and_band()

def prin(X,y,file):
	clf = MLPRegressor(solver='lbfgs',activation='relu',hidden_layer_sizes=(50,100,10,))
	#clf = LinearRegression()
	clf.fit(X[:-10], y[:-10])

	accuracy = clf.score(X[:-10],y[:-10])
	print 'accuracy',accuracy,'\n'
	
	pr=clf.predict(X[-10:])
	print 'Filename                 Percentage Error         Actual Value      Predicted Value           Difference\n'
	for i in range (len(y[-10:])):
		predi=str(round(((pr[i]-y[i])/y[i])*100,2))+' %'
		print file[i]+' '*(20-len(file[i])),' '*(20-len(predi))+ predi, ' '*(20-len(str(y[i])))+str(y[i]) , ' '*(20-len(str(round(pr[i],2))))+str(round(pr[i],2)),' '*(20-len(str(round((y[i]-pr[i]),4))))+str(round((y[i]-pr[i]),4))
	#print 'Mean square Error',mean_squared_error(X,pr)
	#print 'R2 score',r2_score(X,pr)
	#test(X,y,file,clf.coef_[0],clf.intercept_[0])
	return pr 

def amide_fit(data):
	X,y=[],[]
	file=[]
	for i in data:
		file.append(i.split('/')[-1])
		a,b,c,d,e,f,g,h,i,j=data[i]
		li=[a,b,c,d,e,f,g,h,i]
		li=map(lambda x : x[-1],li)
		X+=[li]
		y+=[float(j)]
	prin(X,y,file)

def test(data):
	X,y=[],[]
	file=[]
	m=300
	for i in data:
		file.append(i.split('/')[-1])
		li=data[i][:-1]
		li=map(lambda x : x[-1],li)
		for j in range (len(li),m):
			li+=[0]
		X+=[li]
		y+=[float(data[i][-1])]
	prin(X,y,file)


def calc(data):
	name_c={'-t':test,'-a':amide_fit}
	pka=np.load('/'.join(sys.argv[0].split('/')[:-1])+'/pka.npy').item()
	d=np.load(data).item()
	
	ref=0
	for i in d:
		name=i.split('/')[-1].split('.')[0]
		try:
			d[i]=d[i]+[pka[name]]
		except KeyError:
			ref=1
			pka[name]=float(raw_input('Enter pka for '+i+' : '))
			d[i]=d[i]+[pka[name]]

	if ref:
		if raw_input('Save data : ')=='y':
			np.save('/'.join(sys.argv[0].split('/')[:-1])+'/pka.npy',pka)
	if len(sys.argv)>2:
		name_c[sys.argv[2]](d)
	else:
		test(d)

	#amide_fit(d)
	#make_fit(d)
	#phenol_fit(d)
	#test(d)
	#symm_fit(d)
	#make_excel(d,raw_input('Enter excel name : '),make_fit)

calc(sys.argv[1])
#add_data(sys.argv[1])









