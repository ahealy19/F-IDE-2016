from __future__ import division
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import sys, common, os
import itertools

from subprocess import call
from sklearn import svm, cross_validation, metrics
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

"""
Use a KFold cross-validation process to compare several regression models.
The accuracy of their rankings is evaluated an a big table is produced

Andrew Healy, Aug. 2016
"""

# these parameters were chose by the process decribed at:
# http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

PARAMS = {	'CVC3':{'C': 10.0, 'gamma': 100.0},
			'Z3-4.4.1':{'C': 10.0, 'gamma': 10.0},
			'Z3-4.3.2':{'C': 10.0, 'gamma': 10.0},
			'CVC4':{'C': 10.0, 'gamma': 1.0},
			'Yices':{'C': 100.0, 'gamma': 0.1},
			'veriT':{'C': 10.0, 'gamma': 1.0},
			'Alt-Ergo-1.01':{'C': 10.0, 'gamma': 100.0},
			'Alt-Ergo-0.95.2':{'C': 10.0, 'gamma': 100.0}}

REGRESSORS = [
	('NuSVR', svm.NuSVR()),
	('RandomForestRegressor', RandomForestRegressor(n_estimators=100,random_state=42,min_samples_leaf=5)),
	('Ridge', RidgeCV()),
	('LinearReg', LinearRegression()),
	('K-nn Regressor', KNeighborsRegressor(weights='distance', algorithm='auto')),
	('DecisionTreeRegressor', DecisionTreeRegressor(min_samples_leaf=5,random_state=42)),
]


def get_predict(label, clf, Xtrn, Xtst, ytrn, ytst, use_weights=False, weights=False):

	
	# SVRs need their data to be scaled
	if label in ['NuSVR', 'K-nn Regressor']:
		Xtrn = DataFrame(scaler.transform(Xtrn), columns=Xtrn.columns, index=Xtrn.index)
		Xtst = DataFrame(scaler.transform(Xtst), columns=Xtst.columns, index=Xtst.index)

	if label == 'NuSVR':
		# this one can't deal with multi-output:
		# treat each prover separately and then combine
		predicted = {}
	
		for p in common.PROVERS:
			my_dict = {'C':PARAMS[p]['C'], 'kernel':'rbf', 'gamma':PARAMS[p]['gamma']}
			clf.set_params(**my_dict)
			if use_weights:
				clf.fit(Xtrn, ytrn[p], sample_weight=weights)
			else:
				clf.fit(Xtrn,ytrn[p])
			predicted[p] = clf.predict(Xtst)
	
		predictions = DataFrame(predicted, columns=ytrn.columns, index=Xtst.index)
	else:
		if use_weights and label != 'K-nn Regressor':
				
				clf.fit(Xtrn, ytrn, sample_weight=weights)
				
		elif not use_weights:
			clf.fit(Xtrn, ytrn)
	
		predictions = DataFrame(clf.predict(Xtst), columns=ytrn.columns, index=Xtst.index ).fillna(0)

	score = metrics.r2_score(ytst, predictions, multioutput='uniform_average')

	# return (the predictions, a series of rankings from the predictions, r2 score)
	return (predictions,predictions.apply(common.get_strings, axis=1), score)



def evaluate(preds, actual, df):



	data = DataFrame({'actual' : actual, 'preds': preds, 'key':actual.index}) 

	# cumulative time to a Valid or Invalid result
	data['cum time'] = data.apply(lambda ser: 
		common.time_to_valid( ser['preds'], 
			df.ix[ser['key'], common.TIMES+common.RESULTS ] ), axis=1 )
	# cumulative difference in scores for ranking positions
	data['cum score diff'] = data.apply(lambda ser:
		common.sum_score_diff( ser['actual'], ser['preds'], 
		df.ix[ser['key'], common.PROVERS ] ), axis=1)
	# normalised distributed cumulative gain for scoring rankings 
	data['ndcg'] = 	data.apply(lambda ser: 
		common.ndcg_k( ser['actual'], ser['preds'], len(common.PROVERS) ), axis=1)
	# mean average error
	data['mae'] = data.apply(lambda ser:
		common.mae(ser['actual'], ser['preds']), axis=1)	
	
	return {'ndcg': data['ndcg'].mean(), 		# avg ndcg for all predicted rankings  
			'cum time': data['cum time'].mean(),# avg cumulative time to Valid or Invalid
			'MAE': data['mae'].mean(),			# mean mean average error
			'score diff': data['cum score diff'].mean() # regression error: mean diff.
			}
	


if __name__ == '__main__':

	test = DataFrame.from_csv('whygoal_valid_test.csv').fillna(0)
	# separate input from output
	X = test.drop(common.IGNORE, axis=1)
	y = {}
	for p in common.PROVERS:
		y[p] = test.apply(lambda x:
					# calculate each solver's cost given result, time and timeout 
					#common.new_score_func_single(x[p+' result'], x[p+' time'], 10.0), 
					common.twice_delta_score_func(x[p+' result'], x[p+' time'], 10.0),
					axis=1)
		test[p] = y[p]

	y = DataFrame(y)

	scaler = StandardScaler()
	# use std deviation for weights
	weights = y.apply(lambda ser: ser.std(), axis=1)

	# for descretising values - divide all scores by a value
	y_bin = y.apply(lambda x: np.floor_divide(x, 2.5))

	total = []

	kf = cross_validation.KFold(test.shape[0], n_folds=4, shuffle=True, random_state=42)

	# each fold
	for i, (train_index, test_index) in enumerate(kf):

		X_train, X_test = X.ix[train_index], X.ix[test_index]
		y_train, y_test = y.ix[train_index], y.ix[test_index]

		# scaler fitted only on training data
		scaler = scaler.fit(X_train)

		train_weights = weights.ix[X_train.index]

		y_train_bin = y_bin.ix[y_train.index]

		# the ground truth: 'Best'
		actual = y_test.apply(common.get_strings, axis=1)
		# the reverse of 'Best': 'Worst'
		worst = actual.apply(lambda x: x[::-1])

		models = {label: get_predict(label, clf, X_train, X_test, y_train, y_test)
			for label, clf in REGRESSORS }

		print 'Done no weights, no binning'

		scores = {label: models[label][1]
			for label, _ in REGRESSORS }

		for (label, clf) in REGRESSORS:
			models[label+' bin'] = get_predict(label, clf, X_train, X_test, y_train_bin, y_test)
			scores[label+' bin'] = models[label+' bin'][1]

		print 'Done no weights, binning'

		for (label, clf) in REGRESSORS:
			models[label+ ' weights'] = get_predict(label, clf, X_train, X_test, y_train, y_test, use_weights=True, weights=train_weights.values)
			scores[label+' weights'] = models[label+ ' weights'][1]

		print 'Done weights, no binning'

		for (label, clf) in REGRESSORS:
			models[label+' weights bin'] = get_predict(label, clf, X_train, X_test, y_train_bin, y_test, use_weights=True, weights=train_weights.values)
			scores[label+' weights bin'] = models[label+' weights bin'][1]

		print 'Done weights, binning'

		table = {label : evaluate(clf, actual, test)
			for label, clf in scores.iteritems() }

		table['Random'] =  {'ndcg' : 0.355384223121,
						 'MAE' : 2.625,
						 'cum time': 19.057453462,
						 'score diff': 50.765591532,
						 'score': 0.0
						 # computed by permute_rankings.py
	 					}

		for label, m in models.iteritems():
			table[label]['score'] = m[2]

		table['Worst'] = evaluate(worst, actual, test)
		table['Best'] = evaluate(actual, actual, test)
		table['Best']['score'] = 1.0
		table['Worst']['score'] = 0.0
		
		total.append(DataFrame(table).transpose())
		print('Fold '+str(i+1)+' complete')


	cols = (total[0]).columns
	rows = (total[0]).index

	# average the folds' results and
	# put in a dataframe for printing and latex
	final = {row: {col: np.mean([tbl.ix[row, col] for tbl in total])
					for col in cols}
			for row in rows}
	rv = DataFrame(final).transpose().sort_index()
	
	
	txt = """
\documentclass[]{{article}}
\\usepackage{{booktabs, graphicx}}

\\begin{{document}}

\\begin{{table}}
\caption{{ {title} }}
\\resizebox{{\\textwidth}}{{!}}{{%
{table} }}
\end{{table}}
\end{{document}}

""".format(title='Predictor Selection',
	table=rv.ix[:, [1,2,3,0,4]].to_latex(
		float_format=lambda x: "{0:.2f}".format(x) ) )
	
	with open('compare_regressors.tex', 'w') as f:
		f.write(txt)
	
	command = "pdflatex -synctex=1 -interaction=nonstopmode compare_regressors.tex"
	call(command.split(), shell=False)
	os.remove('compare_regressors.aux')
	os.remove('compare_regressors.log')
	os.remove('compare_regressors.synctex.gz')
	# the first few
	print(rv.head(10))
