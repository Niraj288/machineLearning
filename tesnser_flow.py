import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import itertools
tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["Filename", "Qa1", "Qa2", "Qw", "pKa (e)"]
FEATURES = ["Qa1", "Qa2", "Qw"]
LABEL = "pKa (e)"

training_set = pd.read_csv(sys.argv[1], skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

#test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
 #                      skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv(sys.argv[1], skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
print feature_cols

regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols, hidden_units=[10, 10])


def input_fn(data_set, pred = False):
    
    if pred == False:
        
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        
        return feature_cols, labels

    if pred == True:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        
        return feature_cols

regressor.fit(input_fn=lambda: input_fn(training_set), steps=500)

#ev = regressor.evaluate(input_fn=lambda: input_fn(prediction_set), steps=1)
#print ev
#loss_score = ev["loss"]
#print "Loss: {0:f}".format(loss_score)

y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
#predictions = list(itertools.islice(y, 87378))

print list(y)





'''
def input_fn(data_set, pred = False):
    
    if pred == False:
        
        return data_set[0],data_set[1]

    if pred == True:
        
        return data_set[0]

def nn(X,y):
	tf.logging.set_verbosity(tf.logging.ERROR)
	regressor = tf.contrib.learn.DNNRegressor(feature_columns=[X,y], 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])
	regressor.fit(input_fn=lambda: input_fn([X,y]), steps=10)
	ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
	loss_score1 = ev["loss"]
	print "Final Loss on the testing set: {0:f}".format(loss_score1)
	y = regressor.predict(input_fn=lambda: input_fn(testing_set))
	predictions = list(itertools.islice(y, testing_set.shape[0]))

def make_fit(data):
	X,y=[],[]
	file=[]
	for i in data:
		if len(data[i])<4:
			continue
		print i,data[i]
		file.append(i.split('/')[-1])
		a,b,c,d=data[i]
		a1=(float(a[-1])+float(b[-1]))/2
		b1=float(c[-1]) 
		X+=[[a1**2,b1**2,a1*b1,a1,b1]]
		y+=[[float(d)]]
	nn(X,y)

def calc(data):
	name_c=''#{'-a':amide_fit,'-p':phenol_fit,'-m':make_fit,'-s':symm_fit}
	pka=np.load('pka.npy').item()
	d=np.load(data).item()
	
	ref=0
	for i in d:
		if len(d[i])==4:
			break
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
		make_fit(d)

	#amide_fit(d)
	#make_fit(d)
	#phenol_fit(d)
	#test(d)
	#symm_fit(d)
	#make_excel(d,raw_input('Enter excel name : '),make_fit)

calc(sys.argv[1])
'''
