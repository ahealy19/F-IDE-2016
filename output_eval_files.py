import json, common, itertools
import numpy as np
import pandas as pd
from pandas import DataFrame, Series 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from subprocess import call
import matplotlib.pyplot as plt
import compare_regressors as cr

"""
Outputs a number of files for evaluation

Andrew Healy, Aug. 2016
"""

def tree_from_sklearn(dt, features, outputs):
	"""
	Turning an sklearn tree into a dictionary of my own
	before printing to a json file. The schema is designed to be 
	legible for easy manipulation
	"""
	t = dt.tree_

	nodes = [	{'index': i,
				'type':'node',
				'feature': features[t.feature[i]],
			 	'threshold': t.threshold[i],
			 	'true': t.children_left[i],
			 	'false': t.children_right[i] }
			for i in xrange(t.node_count) ]

	for i, n in enumerate(nodes):
		if n['false'] == -1:
			temp = {}
			temp['type'] = 'leaf'
			temp['index'] = i
			for j in xrange(dt.n_outputs_):
				temp[ outputs[j] ] = t.value[i][j][0]
			nodes[i] = temp
			
	return nodes

def forest_from_sklearn(rf, features, outputs):
	# a forest is just an array of trees....
	return [ tree_from_sklearn(dt, features, outputs)
				for dt in rf.estimators_ ]

train = DataFrame.from_csv('whygoal_valid_test.csv').fillna(0)
rf = RandomForestRegressor(n_estimators=100,random_state=42,min_samples_leaf=5)
X = train.drop(common.IGNORE, axis=1)
y = {}
for p in common.PROVERS:
	y[p] = train.apply(lambda x: 
				common.new_score_func_single(x[p+' result'], x[p+' time'], 10.0), 
				axis=1)
	train[p] = y[p]

y = DataFrame(y)
rf.fit(X, y)


########### feature_importances.txt ##################

important = Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
# attribute to tell how important the input variables are
# to the Random Forest when making decisions
print important
with open('feature_importances.txt','w') as out:
	out.write(str(important))

########################################################

########### forest.json ################################

f = forest_from_sklearn(rf, X.columns, y.columns)

with open('forest.json', 'w') as jfile:
	json.dump(f, jfile, indent=4, separators=(',',': '))
print('forest written')

########################################################

# now for testing stuff

test = DataFrame.from_csv('whygoal_test.csv').fillna(0)

X_test = test.drop(common.IGNORE, axis=1)
y_test = {}

# save the cost predictions for each goal
test = pd.concat([ test, 
					DataFrame(rf.predict(X_test), 
						columns=['Predicted '+c for c in y.columns], 
						index=X_test.index ).fillna(0) ], 
				axis=1)
# and turn them into rankings
predictions = DataFrame(rf.predict(X_test), columns=y.columns, index=X_test.index ).fillna(0)
pred_ranks = predictions.apply(common.get_strings, axis=1)

actual = y_test.apply(common.get_strings, axis=1)
worst = actual.apply(lambda x: x[::-1])

output = {	'Valid':common.is_valid,
			'Invalid':common.is_invalid,
			'Unknown':common.is_unknown,
			'Timeout':common.is_timeout,
			'Failure':common.is_error} 

# count the results for each prover/goal
results = {p : 
			{o: test.apply(lambda ser: fun(ser[p+' result']), axis=1).sum()
			for o, fun in output.iteritems() }  
		for p in common.PROVERS }

# how many of each are there?
print('files: {}\ntheories: {}\ngoals: {}'.format(
	len(test['file'].unique()),
	len(test['theory'].unique()),
	len(test['goal'].unique())
	))

test['key'] = test.index
test['rf'] = pred_ranks
test['best'] = actual
test['worst'] = worst

# add results and times for strategies and where4
for x in ['rf', 'best', 'worst']:

	test[x+' time'] = test.apply(lambda ser: common.time_to_valid( ser[x], 
				test.ix[ser['key'], common.TIMES+common.RESULTS ] ), axis=1 )
	test[x+' result'] = test.apply(lambda ser: common.best_result_from_rank( ser[x], 
				test.ix[ser['key'], common.RESULTS ] ), axis=1 )

# before counting their results
results['Where4'] = {o : test.apply(lambda ser: 
	fun( test.ix[ser['key'], common.REV_MAP[ser['rf'][0]]+' result'] ), axis=1).sum()
	for o, fun in output.iteritems()}

results['Best'] = {o : test.apply(lambda ser: 
	fun( test.ix[ser['key'], common.REV_MAP[ser['best'][0]]+' result'] ), axis=1).sum()
	for o, fun in output.iteritems()}

results['Worst'] = {o : test.apply(lambda ser: 
	fun( test.ix[ser['key'], common.REV_MAP[ser['best'][-1]]+' result'] ), axis=1).sum()
	for o, fun in output.iteritems()}

# this one takes the most common result among the solvers for each goal
results['Random'] = {o: test.apply(lambda ser:
	fun( common.avg_result(ser[common.RESULTS]) ), axis=1).sum()
	for o, fun in output.iteritems()}

############## data_for_second_barchar.csv ###################

results = DataFrame(results)
results.to_csv('data_for_second_barchart.csv')
print results

############## data_for_second_linegraph.csv #################

table = {'Best' : cr.evaluate(actual, actual, test),
		'Worst' : cr.evaluate(worst, actual, test),
		'Where4' : cr.evaluate(pred_ranks, actual, test)}

table['Where4']['score'] = metrics.r2_score(y_test, predictions, multioutput='uniform_average')

s = 'aAzZ34vy'
N = len(s)
all_orders = [''.join(rand) for rand in itertools.permutations(s, r=N)]

# this one takes a while: find the average time for every permutation
# to return a Valid/Invalid answer on the test set

#test['rand time'] = test.apply(lambda ser: np.mean([common.time_to_valid( rand, 
#			test.ix[ser['key'], common.TIMES+common.RESULTS] ) 
#			for rand in all_orders]), axis=1 )


# Be careful! don't want to overwrite the csv file below mistakenly
#test.to_csv('data_for_second_linegraph.csv')
