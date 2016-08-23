import common, os
import numpy as np
from pandas import DataFrame, Series 
import matplotlib.pyplot as plt

"""
Plot the cumulative time taken for the three theoretical strategies and Where
to find an answer to the goals in the test dataset

Andrew Healy, Aug. 2016
"""

test = DataFrame.from_csv('data_for_second_linegraph.csv')
# which files and theories are provable by any solver
f_provable = [u'add_list.mlw', u'algo64.mlw', u'algo65.mlw', u'all_distinct.mlw', u'arm.mlw', u'assigning_meanings_to_programs.mlw', u'balance.mlw', u'binary_multiplication.mlw', u'binary_search.mlw', u'braun_trees.mlw', u'bubble_sort.mlw', u'checking_a_large_routine.mlw', u'conjugate.mlw', u'cursor.mlw', u'division.mlw', u'euler002.mlw', u'ewd673.mlw', u'fact.mlw', u'fib_memo.mlw', u'fill.mlw', u'finger_trees.mlw', u'finite_tarski.mlw', u'flag.mlw', u'foveoos11_challenge1.mlw', u'induction.mlw', u'insertion_sort_naive.mlw', u'lcp.mlw', u'mccarthy.mlw', u'mjrty.mlw', u'muller.mlw', u'relabel.mlw', u'remove_duplicate.mlw', u'remove_duplicate_hash.mlw', u'resizable_array.mlw', u'rightmostbittrick.mlw', u'same_fringe.mlw', u'selection_sort.mlw', u'snapshotable_trees.mlw', u'sorted_list.mlw', u'swap.mlw', u'there_and_back_again.mlw', u'tower_of_hanoi.mlw', u'tree_of_list.mlw', u'vacid_0_build_maze.mlw', u'verifythis_2015_dancing_links.mlw', u'vstte10_aqueue.mlw', u'vstte12_two_way_sort.mlw', u'zeros.mlw']
t_provable = [[u'add_list.mlw', u'AddListImp'], [u'add_list.mlw', u'AddListRec'], [u'add_list.mlw', u'SumList'], [u'algo64.mlw', u'Algo64'], [u'algo65.mlw', u'Algo65'], [u'all_distinct.mlw', u'AllDistinct'], [u'arm.mlw', u'InsertionSortExample'], [u'arm.mlw', u'M'], [u'arm.mlw', u'ARM'], [u'assigning_meanings_to_programs.mlw', u'Division'], [u'assigning_meanings_to_programs.mlw', u'Sum'], [u'bag.mlw', u'Bag'], [u'bag.mlw', u'BagSpec'], [u'bag.mlw', u'ResizableArraySpec'], [u'bag.mlw', u'Harness'], [u'balance.mlw', u'Puzzle8'], [u'balance.mlw', u'Roberval'], [u'balance.mlw', u'Puzzle12'], [u'binary_multiplication.mlw', u'BinaryMultiplication'], [u'binary_search.mlw', u'BinarySearchAnyMidPoint'], [u'binary_search.mlw', u'BinarySearchInt32'], [u'binary_search.mlw', u'BinarySearch'], [u'bitcount.mlw', u'Hamming'], [u'bitvector_examples.mlw', u'Hackers_delight'], [u'bitvector_examples.mlw', u'Test_imperial_violet'], [u'bitvector_examples.mlw', u'Test_from_bitvector_example'], [u'bitvector_examples.mlw', u'Hackers_delight_mod'], [u'braun_trees.mlw', u'BraunHeaps'], [u'bubble_sort.mlw', u'BubbleSort'], [u'checking_a_large_routine.mlw', u'CheckingALargeRoutine'], [u'conjugate.mlw', u'Test'], [u'conjugate.mlw', u'Conjugate'], [u'counting_sort.mlw', u'Spec'], [u'cursor.mlw', u'Cursor'], [u'cursor.mlw', u'TestCursor'], [u'cursor.mlw', u'TestArrayCursor'], [u'cursor.mlw', u'IntListCursor'], [u'cursor.mlw', u'TestListCursor'], [u'cursor.mlw', u'IntArrayCursor'], [u'defunctionalization.mlw', u'Defunctionalization2'], [u'defunctionalization.mlw', u'Expr'], [u'defunctionalization.mlw', u'DirectSem'], [u'division.mlw', u'Division'], [u'dyck.mlw', u'Dyck'], [u'euler001.mlw', u'Euler001'], [u'euler001.mlw', u'SumMultiple'], [u'euler002.mlw', u'FibSumEven'], [u'euler002.mlw', u'Solve'], [u'euler002.mlw', u'FibOnlyEven'], [u'ewd673.mlw', u'EWD673'], [u'fact.mlw', u'FactRecursive'], [u'fact.mlw', u'FactImperative'], [u'fib_memo.mlw', u'FibMemo'], [u'fibonacci.mlw', u'FibRecGhost'], [u'fibonacci.mlw', u'Zeckendorf'], [u'fibonacci.mlw', u'FibonacciLinear'], [u'fibonacci.mlw', u'FibonacciTest'], [u'fibonacci.mlw', u'SmallestFibAbove'], [u'fibonacci.mlw', u'Mat22'], [u'fibonacci.mlw', u'FibRecNoGhost'], [u'fill.mlw', u'Fill'], [u'finger_trees.mlw', u'FingerTrees'], [u'finite_tarski.mlw', u'Tarski'], [u'finite_tarski.mlw', u'Tarski_while'], [u'finite_tarski.mlw', u'Tarski_rec'], [u'flag.mlw', u'Flag'], [u'foveoos11_challenge1.mlw', u'Max'], [u'gcd.mlw', u'EuclideanAlgorithmIterative'], [u'gcd.mlw', u'EuclideanAlgorithm'], [u'gcd.mlw', u'EuclideanAlgorithm31'], [u'hackers-delight.mlw', u'Utils'], [u'hackers-delight.mlw', u'Hackers_delight'], [u'induction.mlw', u'Induction1'], [u'induction.mlw', u'Induction2'], [u'induction.mlw', u'LemmaFunction2'], [u'induction.mlw', u'LemmaFunction3'], [u'induction.mlw', u'LemmaFunction1'], [u'induction.mlw', u'Hyps'], [u'insertion_sort.mlw', u'InsertionSort'], [u'insertion_sort_naive.mlw', u'InsertionSortParam'], [u'insertion_sort_naive.mlw', u'InsertionSortNaive'], [u'insertion_sort_naive.mlw', u'InsertionSortNaiveGen'], [u'insertion_sort_naive.mlw', u'InsertionSortParamBad'], [u'inverse_in_place.mlw', u'Harness'], [u'isqrt.mlw', u'Simple'], [u'isqrt.mlw', u'Square'], [u'lcp.mlw', u'LCP'], [u'linear_probing.mlw', u'HashedTypeWithDummy'], [u'linked_list_rev.mlw', u'Disjoint'], [u'max_matrix.mlw', u'Bitset'], [u'max_matrix.mlw', u'HashTable'], [u'maximum_subarray.mlw', u'Algo4'], [u'maximum_subarray.mlw', u'Algo5'], [u'maximum_subarray.mlw', u'Algo2'], [u'maximum_subarray.mlw', u'Algo1'], [u'maximum_subarray.mlw', u'Spec'], [u'mccarthy.mlw', u'McCarthy91'], [u'mergesort_array.mlw', u'TopDownMergesort'], [u'mergesort_array.mlw', u'Elt'], [u'mergesort_list.mlw', u'EfficientMerge'], [u'mergesort_list.mlw', u'Elt'], [u'mergesort_list.mlw', u'Mergesort'], [u'mjrty.mlw', u'Mjrty'], [u'muller.mlw', u'Muller'], [u'patience.mlw', u'PatienceCode'], [u'patience.mlw', u'PigeonHole'], [u'patience.mlw', u'PatienceFull'], [u'queens.mlw', u'S'], [u'queens.mlw', u'NQueensSetsTermination'], [u'queens.mlw', u'Solution'], [u'queens_bv.mlw', u'BitsSpec'], [u'queens_bv.mlw', u'S'], [u'queens_bv.mlw', u'Solution'], [u'quicksort.mlw', u'Test'], [u'quicksort.mlw', u'Shuffle'], [u'quicksort.mlw', u'QuicksortWithShuffle'], [u'random_access_list.mlw', u'RAL'], [u'random_access_list.mlw', u'RandomAccessListWithSeq'], [u'register_allocation.mlw', u'DWP'], [u'register_allocation.mlw', u'Spec'], [u'relabel.mlw', u'Relabel'], [u'remove_duplicate.mlw', u'Spec'], [u'remove_duplicate.mlw', u'RemoveDuplicateQuadratic'], [u'remove_duplicate_hash.mlw', u'RemoveDuplicate'], [u'remove_duplicate_hash.mlw', u'MutableSet'], [u'remove_duplicate_hash.mlw', u'Spec'], [u'residual.mlw', u'Test'], [u'residual.mlw', u'DFA'], [u'resizable_array.mlw', u'Test'], [u'resizable_array.mlw', u'ResizableArraySpec'], [u'resizable_array.mlw', u'ResizableArrayImplem'], [u'rightmostbittrick.mlw', u'Rmbt'], [u'ropes.mlw', u'Sig'], [u'ropes.mlw', u'String'], [u'same_fringe.mlw', u'Test'], [u'same_fringe.mlw', u'SameFringe'], [u'selection_sort.mlw', u'SelectionSort'], [u'sf.mlw', u'MoreHoareLogic'], [u'skew_heaps.mlw', u'SkewHeaps'], [u'skew_heaps.mlw', u'Heap'], [u'snapshotable_trees.mlw', u'Iterator'], [u'snapshotable_trees.mlw', u'ITree'], [u'snapshotable_trees.mlw', u'Enum'], [u'snapshotable_trees.mlw', u'Tree'], [u'snapshotable_trees.mlw', u'Harness'], [u'snapshotable_trees.mlw', u'BSTree'], [u'sorted_list.mlw', u'FindInSortedList'], [u'sudoku.mlw', u'Test'], [u'sudoku.mlw', u'Grid'], [u'sudoku.mlw', u'TheClassicalSudokuGrid'], [u'swap.mlw', u'Swap'], [u'there_and_back_again.mlw', u'Palindrome'], [u'there_and_back_again.mlw', u'Convolution'], [u'topological_sorting.mlw', u'Graph'], [u'topological_sorting.mlw', u'Online_Basic'], [u'topological_sorting.mlw', u'Online_graph'], [u'tower_of_hanoi.mlw', u'Hanoi'], [u'tower_of_hanoi.mlw', u'Tower_of_Hanoi'], [u'toy_compiler.mlw', u'Expr'], [u'toy_compiler.mlw', u'StackMachine'], [u'tree_of_list.mlw', u'TreeOfList'], [u'unraveling_a_card_trick.mlw', u'GilbreathCardTrick'], [u'vacid_0_build_maze.mlw', u'Graph'], [u'vacid_0_build_maze.mlw', u'UnionFind_pure'], [u'vacid_0_build_maze.mlw', u'BuildMaze'], [u'vacid_0_build_maze.mlw', u'Graph_sig'], [u'vacid_0_build_maze.mlw', u'UnionFind_sig'], [u'vacid_0_red_black_trees.mlw', u'Vacid0'], [u'vacid_0_sparse_array.mlw', u'Harness'], [u'verifythis_2015_dancing_links.mlw', u'DancingLinks'], [u'verifythis_2015_parallel_gcd.mlw', u'ParallelGCD'], [u'verifythis_fm2012_LRS.mlw', u'SuffixArray_test'], [u'verifythis_fm2012_LRS.mlw', u'LCP_test'], [u'verifythis_fm2012_LRS.mlw', u'LCP'], [u'verifythis_fm2012_LRS.mlw', u'SuffixSort_test'], [u'verifythis_fm2012_LRS.mlw', u'LRS_test'], [u'verifythis_fm2012_LRS.mlw', u'SuffixArray'], [u'verifythis_fm2012_treedel.mlw', u'Memory'], [u'vstte10_aqueue.mlw', u'AmortizedQueue'], [u'vstte10_inverting.mlw', u'Test'], [u'vstte10_max_sum.mlw', u'TestCase'], [u'vstte10_max_sum.mlw', u'MaxAndSum'], [u'vstte10_queens.mlw', u'NQueens63'], [u'vstte12_ring_buffer.mlw', u'Harness'], [u'vstte12_ring_buffer.mlw', u'RingBufferSeq'], [u'vstte12_ring_buffer.mlw', u'HarnessSeq'], [u'vstte12_tree_reconstruction.mlw', u'ZipperBasedTermination'], [u'vstte12_two_way_sort.mlw', u'TwoWaySort'], [u'zeros.mlw', u'SetZeros'], [u'zeros.mlw', u'AllZeros']]

total = {}
fig = plt.figure()
fig.set_size_inches(12, 8, forward=True)

ax = fig.add_subplot(1,1,1)

styles = {p : ['-.k','-k','--k', ':k'][i] for i, p in enumerate(['best', 'rf', 'worst', 'rand'])}
label_map = {l:['Best','Where4', 'Worst', 'Random'][i] for i,l in enumerate(['best', 'rf', 'worst', 'rand'])}


for p in ['best', 'rf', 'worst', 'rand']:
	
	label = label_map[p]
	prover = p
	
	# sorted by increasing time taken to return a response
	sorted_times = test[['best result', prover+' time']].sort_values(by=[prover+' time'])
	# find just the Valid/Invalid responses
	good_times = sorted_times.apply(lambda ser: ser['best result'] in ['Valid', 'Invalid'], axis=1)
	# and take these times
	actual_times = sorted_times.loc[good_times]
	
	# incrementing the y axis
	total[p] = Series([x+1 for x in range(actual_times.shape[0])], index=actual_times[prover+' time'])
	ax.plot(total[p].index, total[p].values, styles[p], label=label+': {:.2f} secs'.format(total[p].index[-1]))

	####### print average times ###########
	print 'Goal: '+str(actual_times.mean())
	proved_files = test.apply(lambda ser: ser['file'] in f_provable, axis=1)
	actual_times = sorted_times.loc[proved_files]
	print 'File: '+str(actual_times.mean())

	proved_files = test.apply(lambda ser: ser['theory'] in [x[1] for x in t_provable]
													and ser['file'] in [x[0] for x in t_provable], axis=1)
	actual_times = sorted_times.loc[proved_files]
	print 'Theory: '+str(actual_times.mean())


ax.set_xscale('log',basex=2)
ax.set_ylabel('Number of Valid/Invalid responses')
ax.set_xlabel('Time (log 2 secs)')
ax.legend(loc='lower right', ncol=2)
plt.savefig(os.path.join('paper','line_graph_eval_provers.pdf'), bbox_inches='tight')
fig.show()