import itertools, common
import numpy as np
from pandas import Series

test = {'Alt-Ergo-0.95.2 result':'Unknown', 'Alt-Ergo-1.01 result':'Unknown', 'CVC3 result':'Valid', 'CVC4 result':'Unknown', 'veriT result':'Timeout', 'Yices result': 'Timeout', 'Z3-4.3.2 result':'Timeout', 'Z3-4.4.1 result':'Timeout', 'Alt-Ergo-0.95.2 time':'0.134', 'Alt-Ergo-1.01 time':0.170, 'CVC3 time':0.356, 'CVC4 time':0.173, 'veriT time':10.109, 'Yices time':10.161, 'Z3-4.3.2 time':10.115, 'Z3-4.4.1 time':10.131}

s = sorted('aAzZ34vy')
N = len(s)
all_orders = [''.join(rand) for rand in itertools.permutations(s, r=N)]

ser = Series(test)