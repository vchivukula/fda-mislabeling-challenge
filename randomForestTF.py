#!/usr/bin/env python3

#	Summary:
#	Tensorflow random forest implementation to predict gender & MSI using protein levels
#	Missing values imputed using KNN
#
#	BIOL 8803F
#	Dongjo Ban

import os
import uuid
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut, KFold
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from fancyimpute import KNN, IterativeImputer, SoftImpute, BiScaler, SimpleFill

def randomForestParams():
	return tensor_forest.ForestHParams(
			num_classes=2,num_features=4115,
			num_trees=10, max_nodes=100).fill()

def generateDataset(x_data, y_data, num_splits):
	# Create dataset for k-fold cross-validation
	def gen():
		for train_index, test_index in KFold(num_splits).split(x_data):
			x_train, x_test = x_data[train_index], x_data[test_index]
			y_train, y_test = y_data[train_index], y_data[test_index]
			yield x_train, y_train, x_test, y_test
	
	return tf.data.Dataset.from_generator(gen, (tf.float32,)*4)

def crossValidate(train_x, train_y, model):

	### Training & Cross-validating
	# 1) Make dataset for cross validation (K-Fold)
	# 2) Split train dataset to train & test

	# Generate dataset to perform k-fold cross validation (currently 10-fold)
	dataset = generateDataset(train_x,train_y,10)
	models = dict()
	i = 1
	print('-----------------------------------------------')

	# Perform k-fold cross-validation
	for x_train, y_train, x_test, y_test in tfe.Iterator(dataset):
		
		# Random Forest
		if model == 'rf':
			hparams = randomForestParams()
			clf = random_forest.TensorForestEstimator(hparams)

		# Temporary location where models are stored
		# print(clf.model_dir)
		# Create and fit model
		clf.fit(x = x_train.numpy(), y = y_train.numpy())

		# Make predictions
		pred = list(clf.predict(x=x_test.numpy()))
		# pred_prob = list(y['probabilities'] for y in pred)
		# Make list of predictions
		pred_class = list(y['classes'] for y in pred)

		# Calculate accuracy
		n = len(y_test.numpy())
		class_zip = list(zip(y_test.numpy(), pred_class))
		n_correct = sum(1 for p in class_zip if p[0]==p[1])
		acc = n_correct/n
		models[i] = acc
		print('The accuracy of model #%d is: %f' % (i, acc))
		i += 1

	print('-----------------------------------------------')
	print('The average accuracy of the models is : %f' % (sum(models.values())/len(models.values())))
	print('-----------------------------------------------')

def predict(train_x, train_y, test_x, model):
	if model == 'rf':
		# Train random forest model on all train samples
		hparams = randomForestParams()
		clf = random_forest.TensorForestEstimator(hparams)
		clf.fit(x=train_x, y=train_y)

	# Return list of predictions
	return list(clf.predict(x=test_x))

def getLabels(pred, label):
	if label == 'gender':
		d = {0:'Female', 1:'Male'}
	elif label == 'msi':
		d = {0:'MSI-High', 1:'MSI-Low/MSS'}
	
	pred_class = list(y['classes'] for y in pred)
	return [d.get(n,n) for n in pred_class]

def outResults(filename, labelsGender, labelsMsi):

	j = 0
	f = open(filename, 'w')

	for idx, row in in4.iterrows():
		matchGender = 0
		matchMsi = 0
		matchBoth = 0

		if row['gender'] != labelsGender[j]:
			matchGender = 1
		if row['msi'] != labelsMsi[j]:
			matchMsi = 1
		if matchGender != 0 or matchMsi != 0:
			matchBoth = 1

		f.write(idx + "\t" + row['gender'] + "\t" + labelsGender[j] + "\t" + str(matchGender) + "\t" + 
			row['msi'] + "\t" + labelsMsi[j] + "\t" + str(matchMsi) + "\t" + str(matchBoth) + '\n')

		j += 1
	
	f.close()

def main():
	# Supress deprecation warnings from TensorFlow
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.enable_eager_execution()

	in1 = pd.read_csv('challenge_data/train_pro.tsv', sep='\t')
	in2 = pd.read_csv('challenge_data/train_cli.tsv', sep='\t', index_col='sample')
	mismatch = pd.read_csv('challenge_data/sum_tab_1.csv')

	# Transpose df
	in1 = in1.transpose()

	# Remove proteins with no values
	in1 = in1.drop(columns=['ATP7A','SMPX','TMEM35A'])
	# Temporarily convert df to mat for imputation
	mat1 = in1.select_dtypes(include=[np.float]).values
	imp1 = pd.DataFrame(KNN(k=3).fit_transform(mat1))
	imp1.columns = in1.columns
	imp1.index = in1.index

	# Only include samples that match
	match = list(mismatch.query('mismatch==0').loc[:,'sample'])

	in1_match = imp1.loc[match]
	in2_match = in2.loc[match]

	x_data = in1_match.astype(np.float32).values
	lb = preprocessing.LabelBinarizer()
	y_data_msi = lb.fit_transform(in2_match.loc[:,'msi']).ravel()
	y_data_gender = lb.fit_transform(in2_match.loc[:,'gender']).ravel()

	# Random Forest
	# crossValidate(x_data, y_data_gender, 'rf')

	### Predict gender and MSI
	# 1) Train model on all train data
	# 2) Make predictions on the test data (test_cli.tsv)

	in3 = pd.read_csv('challenge_data/test_pro.tsv', sep='\t')
	in4 = pd.read_csv('challenge_data/test_cli.tsv', sep='\t', index_col='sample')
	in3 = in3.transpose()
	# Remove proteins with no values
	in3 = in3.drop(columns=['ATP7A','SMPX','TMEM35A'])
	inTest = in3.astype(np.float32).values

	# Make predictions (gender) on test data using random forest model
	pred_gender = predict(x_data, y_data_gender, inTest, 'rf')
	labels_gender = getLabels(pred_gender, 'gender')

	# Make predictions (MSI) on test data using random forest model
	pred_msi = predict(x_data, y_data_msi, inTest, 'rf')
	labels_msi = getLabels(pred_msi, 'msi')

	# Final output
	# outResults('randomForestTF_results.tsv', labels_gender, labels_msi)

if __name__ == '__main__':
	main()
