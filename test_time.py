from __future__ import division 
from benchexec.runexecutor import RunExecutor
import os, shutil, sys
from subprocess import check_output
import numpy as np

executor = RunExecutor()

def n_times(confidence_percent, within_percent, sample_sd, sample_mean):
	#from scipy.stats import t

	alpha = (100 - confidence_percent)/100
	e = (within_percent/2)/100

	t_value = 1.895#t.interval(alpha, within_percent)
	#print(t_value)

	n = round( ( (t_value*sample_sd)/(e*sample_mean) )**2 )
	if n == 0:
		print 'actually zero'
		return 1
	return int(n)


def get_vals(cmd_array, times, val):

	vals = [None]*times
	for x in range(times):

		vals[x] = runexec(cmd_array)[val]
		print('({} : {}'.format(x, vals[x]))
	
	return vals	


def mean_and_sd(vals):

	return np.mean(vals), np.std(vals)
	

def runexec(cmd_array):

	global executor

	return executor.execute_run(
		args=cmd_array, 
		output_filename="runexec.out")

def run( folder, theory, goal, prover, conf,
		times = 5, time = 10):

	#rel_file = os.path.join('.', folder, folder+'.mlw')

	cmd = "why3 prove -P {} -t {} -m 1000 -C {} {} -T {}".format(
		prover,
		time,
		conf,
		folder,
		theory
		)	

	cmd_array = cmd.split()
	cmd_array += ['-G', goal]

	vals = get_vals(cmd_array, times, 'cputime')

	sample_mean, sample_sd = mean_and_sd(vals)
	print('sample_mean: {} sample_sd: {}'.format(sample_mean, sample_sd))

	n = n_times(90, 7, sample_sd, sample_mean)
	print(str(n)+' times')
	
	# don't need more timings, return
	if n <= times:
		return sample_mean

	n = min(n-times, 100) #cap it at 100 - life's too short
	extras = get_vals(cmd_array, n, 'cputime')

	mean, _ = mean_and_sd(vals+extras)
	
	return mean

conf = '/home/andrew/.why3.conf'
folder = 'data/edit_distance/edit_distance.mlw'
theory = 'Word'
goal = 'first_last'
prover = sys.argv[1]

time = run(folder, theory, goal, prover, conf)
print time