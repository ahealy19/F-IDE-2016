import numpy as np

PROVERMAP = {'Z3-4.4.1': 'Z', 'Z3-4.3.2':'z', 'CVC4':'4', 'CVC3':'3', 
			'Yices':'y', 'veriT':'v', 'Alt-Ergo-0.95.2':'a', 'Alt-Ergo-1.01':'A'}
REV_MAP = {val:key for key, val in PROVERMAP.iteritems()}
PROVERS = ['Z3-4.4.1', 'Z3-4.3.2','CVC3','CVC4','Yices','veriT','Alt-Ergo-1.01', 'Alt-Ergo-0.95.2']

RESULT_MAP = {'Valid':0, 'Invalid':0, 'Unknown':10, 'Timeout':15}
# anything else: 20

# this one is for (str -> int) classification used by make_val
CLASS_MAP = {'Valid':0, 'Invalid':5, 'Unknown':10, 'Timeout':15}
CLASS_MAP_REV = {val:key for key, val in CLASS_MAP.iteritems()}
# anything else: 20

OUTPUT = list(RESULT_MAP.keys()) + ['Failure']

LEVELS = ['file','theory', 'goal']
TIMES = [p+' time' for p in PROVERS]
RESULTS = [p+' result' for p in PROVERS]
IGNORE = LEVELS+TIMES+RESULTS
Y_COLUMNS = TIMES+RESULTS

WORST_NDCG = 0.43936240961058232

def euc_distance(v, w):
	return np.linalg.norm(np.array(v)-np.array(w))

def score_func_single(result, time):
	"""this particular score function uses the result penalties 
	defined in RESULT_MAP to use as the x axis with (unscaled) time in secs.
	Returning the Euclidean distance to the origin (0,0)"""
	return euc_distance([RESULT_MAP.get(result, 20), time], [0,0])

def score_func(results, times):
	return [score_func_single(r,t) for r,t in zip(results, times)]

def new_score_func_single(result, time, delta):
	if result == 'Unknown':
		return time+delta
	if result in ['Valid','Invalid','Unknown']:
		return time
	return euc_distance([time, delta], [0,0])

def twice_delta_score_func(result, time, delta):
	if result == 'Unknown':
		return time+delta
	if result in ['Valid','Invalid','Unknown']:
		return time
	return time+(delta*2)	
	

def new_score_func(results,times,delta):
	return [new_score_func_single(r,t,delta) for r,t in zip(results,times)]

def get_best(ser):
	"""What is the first prover in the sorted Series?"""
	ser.sort_values(inplace=True)
	return PROVERMAP[ser.index[0]]

def get_strings(ser):
	"""What is the entire ranking of provers in the sorted series?"""
	ser.sort_values(inplace=True)
	return ''.join([PROVERMAP[c] for c in ser.index])

def get_string_threshold(ser, thresh):
	ser.sort_values(inplace=True)
	return ''.join([PROVERMAP[c] for c in ser.index if ser[c] <= thresh ])


def same_or_not(x, y):
	"""convienent for counting in comprehensions"""
	if x == y: return 1
	return 0

def random_rank():
	l = list(PROVERMAP.values())
	np.random.shuffle(l)
	return ''.join(l)

def random_prover():
	return PROVERMAP[np.random.choice(PROVERS)]

def random_result():
	return np.random.choice(OUTPUT)

def provable(ser):
	results = [ser[p+' result'] for p in PROVERS]
	if 'Valid' in results or 'Invalid' in results:
		return 1
	return 0

# counting the number of each result in evaluations

def is_valid(res):
	if res.startswith('Valid'): return 1
	return 0

def is_invalid(res):
	if res.startswith('Invalid'): return 1
	return 0

def is_unknown(res):
	if res.startswith('Unknown'): return 1
	return 0

def is_timeout(res):
	if res.startswith('Timeout'): return 1
	return 0

def is_error(res):
	return abs(is_valid(res) + is_invalid(res) + is_unknown(res) + is_timeout(res) - 1)

def make_val(v):
	return CLASS_MAP.get(v, 20)

def find_position(s, c):
	for i, x in enumerate(s):
		if x == c: return i
	return -1 #not found - prevent weird results

def relevance(y, c):
	#return 2**(len(y)-1) - distance(y, y_pred, c)
	pos = find_position(y, c)
	if pos == -1:
		return 0
	return len(y)+1 - pos

def dcg2(y, y_pred, k=6):
	return sum( [ (2**(relevance(y, c)) - 1)/np.log2(i+2)
					for i, c in enumerate(y_pred[:k]) 
					])

def ndcg(y, y_pred, k=6):
	return dcg2(y, y_pred, k=k) / dcg2(y, y, k=k)

def scale_ndcg_8(x):
	return (x - WORST_NDCG)/(1.0 - WORST_NDCG)

def ndcg_k(y, y_pred, k):
	if k == 8:
		return scale_ndcg_8(ndcg(y, y_pred, k=k))
	return ndcg(y, y_pred, k=k)

def mae(y, y_pred):
	"""mean absolute error for two ranks(encoded as strings)"""
	errors = [abs(i - find_position(y, c)) 
				for i,c in enumerate(y_pred) ]
	return np.mean(errors)

def sum_score_diff(y, y_pred, scores):
	diff = 0
	for i,p in enumerate(y_pred):
		pred = REV_MAP[p]
		actual = REV_MAP[y[i]]
		diff += abs(scores[pred] - scores[actual])
	return diff

def avg_result(results):
	m = results.mode()
	return np.random.permutation(m)[0]

def best_result_from_rank(y_pred, results):
	if len(y_pred) == 0:
		return 'Unknown'
	first = results[ REV_MAP[y_pred[0]] +' result']
	for p in y_pred:
		prover = REV_MAP[p]
		res = results[prover+' result']
		if res in ['Valid', 'Invalid']:
			return res
	return first

def time_to_valid(y_pred, results):
	time = 0
	for p in y_pred:
		prover = REV_MAP[p]
		time += results[prover+' time']
		if results[prover+' result'] in ['Valid', 'Invalid']:
			return time
	return time