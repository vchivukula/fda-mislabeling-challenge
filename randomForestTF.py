#!/usr/bin/env python3

#	Tensorflow Random Forest Implementation
#	BIOL 8803F
#	Author: Dongjo Ban
#
# 	Script outputs directories for the models. These can be used later.

import os
import uuid
import numpy as np
import pandas as pd
import math
# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut, KFold
# TENSORFLOW
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from fancyimpute import KNN, IterativeImputer, SoftImpute, BiScaler, SimpleFill

# Function to create dataset for k-fold cross-validation
def generateDataset(x_data, y_data, num_splits):
	def gen():
		for train_index, test_index in KFold(num_splits).split(x_data):
			x_train, x_test = x_data[train_index], x_data[test_index]
			y_train, y_test = y_data[train_index], y_data[test_index]
			yield x_train, y_train, x_test, y_test
	
	return tf.data.Dataset.from_generator(gen, (tf.float32,)*4)

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

	# Create 4 labels based on 4 different combinations of gender & msi
	# in2['genderMsi'] = in2['gender']+'.'+in2['msi']

	# Only include samples that match
	match = list(mismatch.query('mismatch==0').loc[:,'sample'])

	in1_match = imp1.loc[match]
	in2_match = in2.loc[match]

	x_data = in1_match.astype(np.float32).values
	lb = preprocessing.LabelBinarizer()
	y_data_msi = lb.fit_transform(in2_match.loc[:,'msi']).ravel()
	y_data_gender = lb.fit_transform(in2_match.loc[:,'gender']).ravel()

	# Define parameters for RF
	hparams = tensor_forest.ForestHParams(
		num_classes=2,num_features=4115,
		num_trees=10, max_nodes=100).fill()

	### Training
	# 1) Make dataset for cross validation (K-Fold)
	# 2) Split train dataset to train & test

	# # Generate dataset to perform k-fold cross validation (currently 10-fold)
	# dataset = generateDataset(x_data,y_data,10)
	# models = dict()
	# i = 1
	# print('-----------------------------------------------')

	# # Perform k-fold cross-validation
	# for x_train, y_train, x_test, y_test in tfe.Iterator(dataset):

	# 	# Create uuid for model
	# 	m_name = uuid.uuid4().hex
	# 	# Make directory for model
	# 	os.mkdir(m_name)

	# 	# Create and fit RF model
	# 	clf = random_forest.TensorForestEstimator(hparams, model_dir = m_name)
	# 	clf.fit(x = x_train.numpy(), y = y_train.numpy())

	# 	# Make predictions
	# 	pred = list(clf.predict(x=x_test.numpy()))
	# 	# pred_prob = list(y['probabilities'] for y in pred)
	# 	# Make list of predictions
	# 	pred_class = list(y['classes'] for y in pred)

	# 	# Calculate accuracy
	# 	n = len(y_test.numpy())
	# 	class_zip = list(zip(y_test.numpy(), pred_class))
	# 	n_correct = sum(1 for p in class_zip if p[0]==p[1])
	# 	acc = n_correct/n
	# 	models[m_name] = acc

	# 	print('The accuracy of model #%d is: %f' % (i, acc))
	# 	i += 1

	# print('-----------------------------------------------')
	# print('The average accuracy of the models is : %f' % (sum(models.values())/len(models.values())))
	# print('-----------------------------------------------')
	# print('\nNames of directories containing models:\n')

	# # Print models and their corresponding accuracies
	# for d in models:
	# 	print('%s\t%f' % (d, models[d]))

	### Predict gender and MSI
	# 1) Train model on all train data
	# 2) Make predictions on the test data (test_cli.tsv)

	in3 = pd.read_csv('challenge_data/test_pro.tsv', sep='\t')
	in4 = pd.read_csv('challenge_data/test_cli.tsv', sep='\t', index_col='sample')
	in3 = in3.transpose()
	# Remove proteins with no values
	in3 = in3.drop(columns=['ATP7A','SMPX','TMEM35A'])
	inTest = in3.astype(np.float32).values

	print('Training the model on all train samples:\n')
	clf_gender = random_forest.TensorForestEstimator(hparams, model_dir = 'finalGender')
	clf_gender.fit(x=x_data, y=y_data_gender)

	clf_msi = random_forest.TensorForestEstimator(hparams, model_dir = 'finalMsi')
	clf_msi.fit(x=x_data, y=y_data_msi)

	pred_gender = list(clf_gender.predict(x=inTest))
	# Make list of predictions
	pred_gender_class = list(y['classes'] for y in pred_gender)
	dictGender = {0:'Female', 1:'Male'}
	labelsGender = [dictGender.get(n,n) for n in pred_gender_class]

	pred_msi = list(clf_msi.predict(x=inTest))
	# Make list of predictions
	pred_msi_class = list(y['classes'] for y in pred_msi)
	dictMsi = {0:'MSI-High', 1:'MSI-Low/MSS'}
	labelsMsi = [dictMsi.get(n,n) for n in pred_msi_class]

	j = 0
	f = open('randomForestTF_results.tsv', 'w')

	for idx, row in in4.iterrows():
		matchGender = 0
		matchMsi = 0
		matchBoth = 0

		if row['gender'] != labelsGender[j]:
			matchGender = 1
		if row['msi'] != labelsMsi[j]:
			matchMsi = 1
		if matchGender != 0 || matchMsi != 0:
			matchBoth = 1

		f.write(idx + "\t" + row['gender'] + "\t" + labelsGender[j] + "\t" + str(matchGender) + "\t" + 
			row['msi'] + "\t" + labelsMsi[j] + "\t" + str(matchMsi) + "\t" + str(matchBoth))
	
	f.close()

if __name__ == '__main__':
	main()