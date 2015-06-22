import numpy as np
from sklearn import datasets, metrics, cross_validation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class EnsembleClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, layer1_classifiers=None, layer2_classifier=None, holdout_size=0.2):
		self.layer1_classifiers	= layer1_classifiers
		self.layer2_classifier = layer2_classifier
		self.holdout_size = holdout_size

	def fit(self, X, y):
		#Split data
		rs = cross_validation.ShuffleSplit(X.shape[0], n_iter=1, test_size=self.holdout_size, random_state=0)
		#Train Classifier
		for train_index, test_index in rs:
			#Train Layer 1 classifiers
			for classifier in self.layer1_classifiers:
				classifier.fit(X[train_index], y[train_index])
			#Get probabilities for holdout set
			preds = np.array([], dtype=np.float64).reshape(len(test_index), 0)
			for classifier in self.layer1_classifiers:
				preds = np.concatenate((preds, classifier.predict_proba(X[test_index])[:,:-1]), axis=1)
			self.layer2_classifier.fit(preds, y[test_index])

	def predict(self, X):
		preds = np.array([], dtype=np.float64).reshape(X.shape[0], 0)
		for classifier in self.layer1_classifiers:
			preds = np.hstack((preds, classifier.predict_proba(X)[:,:-1]))
		return self.layer2_classifier.predict(preds)




digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

rf = RandomForestClassifier()
svm_1 = SVC(gamma=0.001, probability=True)

classifier = EnsembleClassifier(layer1_classifiers=[rf, svm_1], layer2_classifier=KNeighborsClassifier())
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
predicted = classifier.predict(data[n_samples / 2:])

print predicted
print '-------------------'
print digits.target[n_samples / 2:]




