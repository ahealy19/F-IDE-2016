import itertools, common
import numpy as np
from pandas import DataFrame 
from sklearn import metrics

"""
This file is for getting the 'Random' scores with are averaged scores
for each ranking possible - ie. each permutation of the solvers. With 8 solvers, 
there are 8! rankings to try.

As such it can take a long time to run!!!

Andrew Healy, Aug. 2016

"""

# change this to 'whygoal_test_valid.csv' to get the values used in
# compare_regressors.py
test = DataFrame.from_csv('whygoal_test.csv').fillna(0)
s = sorted('aAzZ34vy')
N = len(s)
all_orders = [''.join(rand) for rand in itertools.permutations(s, r=N)]

worst = (1.0, s)

all_scores = [(common.ndcg_k(s, rand, N), rand) for rand in all_orders]
sorted_scores = sorted(all_orders, key=lambda x:x[0])
worst = sorted_scores[-1]
avg = np.mean([x[0] for x in all_scores])

mean_MAE = np.mean([common.mae(s, rand) for rand in all_orders])
mean_score = np.mean([common.same_or_not(s[0], rand[0]) for rand in all_orders])

print ('worst ndcg_k: {}'.format(worst))
print ('avg ndcg_k: {}'.format(avg))
print ('avg MAE: {}'.format(mean_MAE))

for p in common.PROVERS:
	test[p] = test.apply(lambda x: 
				common.new_score_func_single(x[p+' result'], x[p+' time'], 10.0), 
				axis=1)

best_r2_score = metrics.r2_score(test[p], test[p], multioutput='uniform_average')
print('best r2: {}'.format(best_r2_score) )

avg_time_to valid = test[].apply(lambda ser: 
	np.mean([common.time_to_valid(rand, ser) for rand in all_orders]), axis=1 )

# LONG!
# the average time for a Valid/Invalid answer to be returned for every ranking
avg_cum_time = np.mean([ 
			test.apply(lambda ser: common.time_to_valid(rand, ser), axis=1).mean()
			for rand in all_orders ])

print('avg_cum_time: {}'.format(avg_cum_time))

test['rank'] = test[common.PROVERS].apply(common.get_strings, axis=1)

# LONG!
# the average regression error for every ranking
avg_score_diff = np.mean([
			test.apply(lambda ser: common.sum_score_diff(ser['rank'], rand, ser), axis=1).mean()
			for rand in all_orders])

print('avg_score_diff: {}'.format(avg_score_diff))
