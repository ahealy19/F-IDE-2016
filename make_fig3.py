import json, os, common
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import numpy as np

"""
see how long it takes for each prover to return their useful results.
Plot the cumulative number of these results as the timeout increases. 
"""

df = DataFrame.from_csv('whygoal_stats.csv')
provers = common.PROVERS

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(1,1,1)
x = 0

styles = {p : ['-k',':k','--k','-.k','-c','--c','-.c',':c'][i] for i, p in enumerate(provers)}

here = os.getcwd()

new_json = [ folder_name+'.json' for folder_name in 
				os.listdir('data') if os.path.isdir(os.path.join('data', folder_name)) 
				]

df2 = {}
# 5,10 covered by the whygoal_stats data
for time in [15, 20, 25, 30, 60]:
	df2[time] = {}
	for p in provers:

		df2[time][p] = 0

		for jfile in sorted(new_json):
			folder_name = jfile[:-5]
			with open(os.path.join('data',folder_name, jfile)) as jf:
				d = json.load(jf)
			for _, fd in d.iteritems():
				for _, td in fd.iteritems():
					for _, gd in td.iteritems():
						if gd[p][str(time)+'secs'] in ['Valid', 'Invalid', 'Unknown']:
							df2[time][p] += 1

		

df2 = DataFrame(df2)

total = {}
fig = plt.figure()
fig.set_size_inches(8, 4, forward=True)
ax = fig.add_subplot(1,1,1)
percentiles = {}

for p in provers:
	
	# plotting up to 10secs
	sorted_times = df[[p+' result', p+' time']].sort_values(by=[p+' time'])
	good_times = sorted_times.apply(lambda ser: ser[p+' result'] in ['Valid', 'Invalid', 'Unknown'], axis=1)
	# just use useful results
	actual_times = sorted_times.loc[good_times]
	# times are the index
	times = Series([x+1 for x in range(actual_times.shape[0])], index=actual_times[p+' time'])
	total[p] = times.append(df2.ix[p])
	# 99th percentile of values
	percentile99 = total[p].index[round(total[p].quantile(0.99))-1]
	print p +' : '+ str(percentile99) + ' ('+str(total[p].values[-1])+')'
	# uncomment this line to draw red lines for the 99 percentiles
	#ax.plot((percentile99,percentile99),(0,total[p].quantile(0.99)),'r-')
	ax.plot(total[p].index, total[p].values, styles[p],  label=p)

	

	if p == 'Z3-4.4.1':
		num1, num2 = 0, 0
		for i in times.index:
			if num1 <= 0.125:
				num1 = i
			if num2 <= 0.25:
				num2 = i
		print times[num1]
		print times[num2]	

# the dotted vertical line
ax.plot((10,10),(0,800),'k:')
# to show more detail for very fast results
ax.set_xscale('log',basex=2)
ax.set_xlabel('Time (log 2 secs)')
ax.set_ylabel('Number of Valid/Invalid/Unknown responses')
ax.legend(loc='best', ncol=2)
plt.savefig(os.path.join('paper','line-graph.pdf'), bbox_inches='tight')
fig.show()

#DataFrame(total).plot(0)