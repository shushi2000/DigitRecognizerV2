#By Segovia on 11/16/2015
import time
import numpy
start_time = time.time()
from numpy import genfromtxt
train = genfromtxt("train.csv", delimiter=",", skip_header=1)
test = genfromtxt("test.csv", delimiter=",", skip_header=1)
print "data loaded..."
train_label = train[:,0]
train_feature = train[:, 1:785]

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

clf = OneVsOneClassifier(LinearSVC(random_state=0))
print "training model..."
clf.fit(train_feature,train_label)

prediction = clf.predict(test)


print "saving output..."
numpy.savetxt("prediction_sklearn.csv", prediction, delimiter=",", fmt="%s")
end_time = time.time();
t = end_time - start_time
print "It took me %d seconds! " %(int(t))