from __future__ import division
import os, sys, json, common, shutil
from pandas import DataFrame, Series
import numpy as np

"""
read the json files. print figures for table 1. save csv file for fig 1

Andrew Healy Aug. 2016
"""

res = '10secs'
time = 'time10'
provers = common.PROVERS
good_result = ['Valid', 'Invalid']

######################
# these 3 methods are for calulating the theoretical prover stats

def goal_provable(goaldict):
    global provers, res, time, good_result

    results = [goaldict[p][time] for p in provers
                if goaldict[p][res] in good_result]

    if len(results) == 0:
        return -1 # goal not provable

    return min(results)

def theory_provable(theorydict):
    if len(theorydict) == 0: # empty theories are trivial
        return 0
    rv = 0 # time taken
    gps = 0 # n goals proved
    for _, goaldict in theorydict.iteritems():
        gp = goal_provable(goaldict)
        if gp >= 0:
            gps += 1
            rv += gp
    if gps == len(theorydict): # all goals have been proved
        return rv # return the time it took
    return -1

def file_provable(filedict):
    if len(filedict) == 0:
        return 0 # empty files are trivial
    rv = 0 # time taken
    tps = 0 # n theories proved
    for _, theorydict in filedict.iteritems():
        tp = theory_provable(theorydict)
        if tp >= 0:
            tps += 1
            rv += tp
    if tps == len(filedict): # all theories have been proved
        return rv # return the time it took
    return -1

########################
# and these show how many are unique to each prover

def goal_provable_by(goaldict):
    global provers, res, time, good_result

    results = [p for p in provers
                if goaldict[p][res] in good_result]

    return set(results)

def theory_provable_by(theorydict):
    
    gps = set(provers) # n goals proved
    for _, goaldict in theorydict.iteritems():
        gp = goal_provable_by(goaldict)
        gps = gps.intersection(gp)
    return gps # may be an empty set


def file_provable_by(filedict):
    
    tps = set(provers) # n theories proved
    for _, theorydict in filedict.iteritems():
        tp = theory_provable_by(theorydict)
        if tp != -1:
            tps = tps.intersection(tp)
    return tps # may be an empty set


########################
here = os.getcwd()
jsfiles = [ folder_name+'.json' for folder_name in 
                os.listdir('data') if os.path.isdir(os.path.join(here, 'data')) ]

f_t_g = ['Theory', 'File', 'Goal']

messages = ['Valid', 'Invalid', 'Timeout', 'Unknown', 'Failure']
total_results_per_prover = {p: {m : 0 for m in messages} 
                                for p in provers}
file_theory_goals_valid_invalid = {p : { t: [0,0]
                                for t in  f_t_g}
                                for p in provers }

portfolio_can = {t : [0,0] for t in f_t_g}
only_p_can = {p: {t: 0 for t in f_t_g } for p in provers}
all_p_can = {t: 0 for t in f_t_g}
no_p_can = {t: 0 for t in f_t_g}
some_ps_can = {t: 0 for t in f_t_g}
n_theories = 0

# iterate through all the json result files
for fle in jsfiles:

    folder_name = fle[:-5]
    with open(os.path.join(here, 'data', folder_name, fle)) as jf:
        d = json.load(jf)

    for f, fd in d.iteritems():
        n_theories += len(fd)

        fp = file_provable(fd)
        if fp >= 0:
            portfolio_can['File'][0] += 1
            portfolio_can['File'][1] += fp
        
        fp = file_provable_by(fd)
        
        n_fp = len(fp)
        if n_fp == 0:
            no_p_can['File'] += 1
        elif n_fp == len(provers):
            all_p_can['File'] += 1
        elif n_fp == 1:
            only_p_can[fp.pop()]['File'] += 1
        else:
            some_ps_can['File'] += 1

        for t, td in fd.iteritems():

            tp = theory_provable(td)
            if tp >= 0:
                portfolio_can['Theory'][0] += 1
                portfolio_can['Theory'][1] += tp

            tp = theory_provable_by(td)
            
            n_tp = len(tp)
            if n_tp == 0:
                no_p_can['Theory'] += 1
            elif n_tp == len(provers):
                all_p_can['Theory'] += 1
            elif n_tp == 1:
                only_p_can[tp.pop()]['Theory'] += 1
            else:
                some_ps_can['Theory'] += 1
                

            for g, gd in td.iteritems():

                gp = goal_provable(gd)
                if gp >= 0:
                    portfolio_can['Goal'][0] += 1
                    portfolio_can['Goal'][1] += gp

                gp = goal_provable_by(gd)
               
                n_gp = len(gp)
                if n_gp == 0:
                    no_p_can['Goal'] += 1
                elif n_gp == len(provers):
                    all_p_can['Goal'] += 1
                elif n_gp == 1:
                    only_p_can[gp.pop()]['Goal'] += 1
                else:
                    some_ps_can['Goal'] += 1



    for prover in provers:
        proved_files = 0
        for filename, filedict in d.iteritems():
            proved_theories = 0
            file_time = 0
            for theoryname, theorydict in filedict.iteritems():
                proved_goals = 0
                theory_time = 0
                for goalname, goaldict in theorydict.iteritems():

                    result = goaldict[prover][res]
                    thistime = goaldict[prover][time]
                    theory_time += thistime # add the goal time to the running theory time

                    if result in good_result:  # the goal has been proved by prover
                        proved_goals += 1
                        file_theory_goals_valid_invalid[prover]['Goal'][0] += 1
                        file_theory_goals_valid_invalid[prover]['Goal'][1] += thistime 
                    
                    # fig 1 counts the results of the 60 second timeout
                    total_results_per_prover[prover][ goaldict[prover]['60secs'] ] += 1
                 
                file_time += theory_time # add the theory time to the running file time

                if proved_goals == len(theorydict):  # the theory has been proved by prover
                    proved_theories += 1
                    file_theory_goals_valid_invalid[prover]['Theory'][0] += 1
                    file_theory_goals_valid_invalid[prover]['Theory'][1] += theory_time

            if proved_theories == len(filedict): # the file has been proved by prover
                proved_files += 1
                file_theory_goals_valid_invalid[prover]['File'][0] += 1
                file_theory_goals_valid_invalid[prover]['File'][1] += file_time

total = DataFrame(total_results_per_prover).transpose()
q = {p: 
        { t : (v[0],v[1]/v[0]) 
          for t, v in td.iteritems() } 
        for p, td in file_theory_goals_valid_invalid.iteritems() 
    }

print 'only_p_can'
print DataFrame(only_p_can)
print 'no_p_can'
print Series(no_p_can)
print 'all_p_can'
print Series(all_p_can)
print 'at least two can'
print Series(some_ps_can)

totals = {t : (sum([only_p_can[p][t] for p in provers ])+no_p_can[t]+all_p_can[t]+some_ps_can[t]) for t in f_t_g}
print 'total'
print Series(totals)

# add results for theoretical solver
q['TS'] = { t : (v[0],v[1]/v[0]) for t,v in portfolio_can.iteritems() }
# save for fig 1 rendering
total.to_csv('fig1_data.csv')

print total
print DataFrame(q).transpose()
# this is how I count theories
print 'There are '+str(n_theories)+' theories'






