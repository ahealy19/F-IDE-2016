import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

"""
read a csv file and render a stacked bar chart showing each prover's
results for 60 second timeout

Andrew Healy, Aug. 2016
"""

df = DataFrame.from_csv('fig1_data.csv')

provers = df.index
N = len(provers)

valids = list(df['Valid'])
unknown = list(df['Unknown'])
timeout = list(df['Timeout'])
failure = list(df['Failure'])

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars

p1 = plt.bar(ind, valids, width, color='1.0')
p2 = plt.bar(ind, unknown, width, color='0.55',
             bottom=valids)

bottom = [unknown[i]+valids[i] for i in xrange(N)]

p3 = plt.bar(ind, timeout, width, bottom=bottom, color='0.8')

bottom = [bottom[i]+timeout[i] for i in xrange(N)]

p4 = plt.bar(ind, failure, width, bottom=bottom, color='0.3')

plt.ylabel('Number of proof obligations')
plt.xticks(ind, provers, rotation = 30)
plt.yticks(np.arange(0, df.ix[0].sum(), 100))
plt.legend((p1[0], p2[0], p3[0], p4[0]), 
	('Valid', 'Unknown', 'Timeout', 'Failure'), 
	loc='upper center', ncol=4,
	 bbox_to_anchor=(0.5, 1.05))
ind = np.arange(N)

# to stack them, change the y in xy

for i,v in enumerate(valids):
	plt.annotate(str(v), xy=(ind[i]+width+0.05,v/2.-0.5))

for i,u in enumerate(unknown):
	plt.annotate(str(u), xy=(ind[i]+width+0.05,valids[i]+u/2.-0.5))

for i,t in enumerate(timeout):
	plt.annotate(str(t), xy=(ind[i]+width+0.05,valids[i]+unknown[i]+t/2.-0.5))

for i,f in enumerate(failure):
	plt.annotate(str(f), xy=(ind[i]+width+0.05,valids[i]+unknown[i]+timeout[i]+f/2.-0.5))

plt.savefig('paper/barcharts.pdf', bbox_inches='tight')