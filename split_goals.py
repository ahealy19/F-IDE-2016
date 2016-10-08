from __future__ import division 
import os, shutil, json, itertools, sys, re
from subprocess import call, check_output, CalledProcessError
import numpy as np

def safedir(f):
	if f in os.listdir(os.getcwd()):
		return False
	return True


if __name__ == '__main__':

	os.chdir('data')
	total = 0

	for folder in sorted([f for f in os.listdir(os.path.join(os.getcwd()))
							if os.path.isdir(os.path.join(os.getcwd(), f))]):


		os.chdir(folder)
		filename = '.'.join([folder, 'mlw'])
		if safedir('split'):
			os.mkdir('split')

		command = 'why3 prove -D /home/andrew/Documents/why3-0.87.1/drivers/why3.drv -o split -a split_goal_wp {}'.format(
			filename)

		call(command.split(), shell=False)
		n = len(os.listdir('split'))
		total += n

		print '{} produced {}'.format(folder, n)

		os.chdir('../')

	print 'total goals: {}'.format(total)

