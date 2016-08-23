import pandas as pd
from pandas import DataFrame
import common, sys,os
import matplotlib.pyplot as plt 

"""
parameterise Where4's performance by using a threshold.
These plots show the effect of the threshold on the time taken for a response(top) 
and number of goals which can be proved (bottom)

Andrew Healy, Aug. 2016
"""

test = DataFrame.from_csv('data_for_second_linegraph.csv')

def make_triple(ser, thresh):
	"""Where's (predicted ranking, result, time) given a goal and threshold"""
	rank = ''.join([ j for j in ser['rf']
		if ser['Predicted '+common.REV_MAP[j]] <= thresh])
	return (rank, 
		common.best_result_from_rank(rank, ser), 
		common.time_to_valid(rank, ser))

def find_intersection(val, arr):
	"""	The x locations for the horizontal line segments"""
	lower, upper = -1, 0
	for a in arr:
		if a > val:
			return lower, upper
		lower += 1
		upper += 1
	return lower, upper

# add the effect of the threshold i to the data
test = pd.concat([DataFrame([ test.apply(lambda ser:
		make_triple(ser, i),
	axis=1)
	for i in xrange(21) ]).T, test], axis=1)

# get the ranks and add to dataframe
ranks = DataFrame({'ranks '+str(i): test.apply(lambda ser: 
		''.join([j for j in ser['rf']
			if ser['Predicted '+common.REV_MAP[j]] <= i ]), axis=1)
		for i in xrange(21) })

test = pd.concat([test, ranks], axis=1)


styles = {p : ['lightpink','orchid','lightsteelblue','cyan','springgreen',
				'olive','orange','red'][i] for i, p in enumerate(common.PROVERS)}

# what are the results at each threshold?
results = DataFrame({'results '+str(i): test.apply(lambda ser:
		common.best_result_from_rank(ser['ranks '+str(i)], ser), axis=1)
		for i in xrange(21) })

# and how long does each take?
times = DataFrame({'times '+str(i): test.apply(lambda ser:
		common.time_to_valid(ser['ranks '+str(i)], ser), axis=1)
		for i in xrange(21)})

# add all these to the dataframe
test = pd.concat([test, results, times], axis=1)

fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True)
fig.subplots_adjust(hspace=0.5)

# the top plot is time
ax = axes[0]
ax.set_xlabel('Cost threshold for Where4')
# the average time for all goals in test at each threshold
ax.plot(range(21), [test['times '+str(i)].mean() for i in range(21)], '-k', label='Where4')

for p in common.PROVERS:
	# the time that each prover took on average
	t = test[p+' time'].mean()
	# draw line segment
	ax.plot(find_intersection(t, [test['times '+str(i)].mean() for i in range(21)]) ,(t,t) , styles[p], label=p)

ax.set_ylabel('Time for Valid/Invalid\n response (secs)')

# bottom plot is number proved
ax = axes[1]
ax.plot( range(21), [len( test.loc[ 
		test.apply(lambda ser: ser['results '+str(i)] in ['Valid', 'Invalid'], axis=1) ] ) 
	for i in range(21) ], '-k', label='Where4' )


for p in common.PROVERS:
	# sort by time
	#sorted_times = test[[p+' result', p+' time']].sort_values(by=[p+' time'])
	# the proved goals
	good_times = test.apply(lambda ser: ser[p+' result'] in ['Valid', 'Invalid'], axis=1)
	# take the proved ones
	actual_times = test.loc[good_times]
	# plot how many there are
	ax.plot(find_intersection(actual_times.shape[0], [len( test.loc[ 
		test.apply(lambda ser: ser['results '+str(i)] in ['Valid', 'Invalid'], axis=1) ] ) 
	for i in range(21) ]), (actual_times.shape[0], actual_times.shape[0]), styles[p] ,label=p )

ax.set_ylabel('Number of Valid/\nInvalid responses')
ax.set_xlabel('Cost threshold for Where4')
ax.set_xticks([x*2 for x in range(11)])
ax.legend( loc='upper center', ncol=3,
	 bbox_to_anchor=(0.5, 3.0))
plt.savefig(os.path.join('paper','thresholds.pdf'), bbox_inches='tight')


