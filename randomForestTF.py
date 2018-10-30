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
	y_data = lb.fit_transform(in2_match.loc[:,'gender']).ravel()


	# Generate dataset to perform k-fold cross validation (currently 10-fold)
	dataset = generateDataset(x_data,y_data,10)
	models = dict()
	i = 1
	print('-----------------------------------------------')

	# Perform k-fold cross-validation
	for x_train, y_train, x_test, y_test in tfe.Iterator(dataset):

		# Create uuid for model
		m_name = uuid.uuid4().hex
		# Make directory for model
		os.mkdir(m_name)

		# Define parameters for RF
		hparams = tensor_forest.ForestHParams(
			num_classes=2,num_features=4115,
			num_trees=10, max_nodes=100).fill()

		# Create and fit RF model
		clf = random_forest.TensorForestEstimator(hparams, model_dir = m_name)
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
		models[m_name] = acc

		print('The accuracy of model #%d is: %f' % (i, acc))
		i += 1

	print('-----------------------------------------------')
	print('The average accuracy of the models is : %f' % (sum(models.values())/len(models.values())))
	print('-----------------------------------------------')
	print('\nNames of directories containing models:\n')

	# Print models and their corresponding accuracies
	for d in models:
		print('%s\t%f' % (d, models[d]))

if __name__ == '__main__':

	main()