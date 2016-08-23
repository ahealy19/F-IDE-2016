import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import os

"""
plots the results for each solver and strategy on 
the test set as a stacked barchart

Andrew Healy, Aug. 2016
"""

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)

df = DataFrame.from_csv('data_for_second_barchart.csv')

provers = ['Alt-Ergo-0.95.2', 'Alt-Ergo-1.01', 'CVC3', 'CVC4', 
			'veriT', 'Yices', 'Z3-4.3.2', 'Z3-4.4.1',
			'Best','Random','Worst','Where4']

df = df.reindex(columns=provers)

N = len(provers)

valids = list(df.ix['Valid'])
unknown = list(df.ix['Unknown'])
timeout = list(df.ix['Timeout'])
failure = list(df.ix['Failure'])

ind = np.arange(N)   # the x locations for the groups
offset = lambda x: 1 if x > 7 else 0
for i,_ in enumerate(ind):
	ind[i] += offset(i) # x offset for strategies and Where4
width = 0.35      # the width of the bars

p1 = ax.bar(ind, valids, width, color='1.0')
p2 = ax.bar(ind, unknown, width, color='0.55',
             bottom=valids)

bottom = [unknown[i]+valids[i] for i in xrange(N)]

p3 = ax.bar(ind, timeout, width, bottom=bottom, color='0.8')

bottom = [bottom[i]+timeout[i] for i in xrange(N)]

p4 = ax.bar(ind, failure, width, bottom=bottom, color='0.3')

ax.set_ylabel('Number of proof obligations')
ax.set_xticks(ind)
ax.set_xticklabels(provers, rotation = 30)
ax.set_yticks(np.arange(0, 263, 50))
ax.legend((p1[0], p2[0], p3[0], p4[0]), 
	('Valid', 'Unknown', 'Timeout', 'Failure'), 
	loc='upper center', ncol=4,
	 bbox_to_anchor=(0.5, 1.05))
ind = np.arange(N)



for i,v in enumerate(valids):
	plt.annotate(str(v), xy=(ind[i]+width+0.05+offset(i),v/2.-0.5))

for i,u in enumerate(unknown):
	plt.annotate(str(u), xy=(ind[i]+width+0.05+offset(i),valids[i]+u/2.-0.5))

for i,t in enumerate(timeout):
	plt.annotate(str(t), xy=(ind[i]+width+0.05+offset(i),valids[i]+unknown[i]+t/2.-0.5))

for i,f in enumerate(failure):
	plt.annotate(str(f), xy=(ind[i]+width+0.05+offset(i),valids[i]+unknown[i]+timeout[i]+f/2.-0.5))

plt.savefig(os.path.join('paper','barcharts2.pdf'), bbox_inches='tight')