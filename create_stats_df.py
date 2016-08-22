from pandas import Series, DataFrame 
import os, json, common

"""
Iterate though the json data files, save the results and times for each prover.
Add these to the goal syntax metrics and save as a csv file for later.

Andrew Healy, Aug. 2016 
"""

provers = common.PROVERS

here = os.getcwd()
folders = sorted([f for f in os.listdir(os.path.join(here,'data')) 
	if os.path.isdir(os.path.join(here,'data', f))
	])

bigd = {}
for folder in folders:

	with open(os.path.join('data',folder, folder+'.json')) as jfile:
		timings_json = json.load(jfile)

	timings = timings_json[folder+'.mlw']

	with open(os.path.join('data',folder,'stats.json')) as jfile:
		st = json.load(jfile)

	for g in st:

		theory = g['theory']
		goal = g['goal']
		g['file'] = folder+'.mlw'

		for goalname in timings[theory]:
			if goalname.endswith(goal):

				#per-prover scores
				for p in provers:
					try:
						g[p+' result'] = timings[theory][goalname][p]['10secs']
					except KeyError:
						print 'Couldn\'t find '+folder+':'+theory+':'+goalname+':'+p+':10secs'
					try:
						g[p+' time'] = timings[theory][goalname][p]['time10']
					except KeyError:
						print 'Couldn\'t find '+folder+':'+theory+':'+goalname+':'+p+':time'
					
		bigd[folder+':'+theory+':'+goal] = Series(g)

df = DataFrame(bigd).transpose()
df.to_csv('whygoal_stats.csv')

