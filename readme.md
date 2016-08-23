# Predicting SMT Solver Performance for Software Verification
## Andrew Healy, Rosemary Monahan, James, F. Power
### [Principles of Programming](http://www.cs.nuim.ie/research/pop/) Research Group, Dept. of Computer Science, Maynooth University, Ireland

### [F-IDE 2016](https://sites.google.com/site/fideworkshop2016/) support data

This repository contains data measuring 8 SMT solvers' performance on the
[Why3](http://why3.lri.fr/) examples dataset. We record the result returned by [Alt-Ergo](https://alt-ergo.ocamlpro.com/) (versions 0.95.2 and 1.01), [CVC3](http://www.cs.nyu.edu/acsys/cvc3/), [CVC4](http://cvc4.cs.nyu.edu/web/), [veriT](http://www.verit-solver.org/), [Yices](http://yices.csl.sri.com/), and [Z3](https://github.com/Z3Prover/z3) (versions 4.3.2 and 4.4.1). We also measure the time taken
by the solver to return the result.

Python libraries we use: [Pandas](http://pandas.pydata.org/), [Numpy](http://www.numpy.org/), [Sci-kit Learn](http://scikit-learn.org/dev/index.html), [Matplotlib](http://matplotlib.org/). All Python files can be run on the command line in the usual way: eg `python <filename.py>`

#### `paper/`
Folder containing latex source files and images for the paper itself

#### `data/`
This folder contains a subfolder for each file in the examples repository. Each folder contains:
 - `<name>.mlw` the WhyML file sent to Why3
 - `<name>.json` a JSON dictionary containing timings and results for various timeout values
 - `stats.json` the syntacic features statically extracted from `<name>.mlw`  (used as independent variables for prediction)

#### `common.py`
A collection of short, commonly-used constants and functions used by many of the other Python scripts.

#### `collect_data_fig1_table1.py`
Python script to collect data from the JSON files. Results printed for Table 1 and saved to `fig1_data.csv` to be read in by `make_fig1.py`

#### `make_fig1.py`
Make the first figure (stacked barcharts - 60 second timeout). Uses `fig1_data.csv`. Renders `barcharts.pdf` to `paper` folder

#### `create_stats_df.py`
Collect data from the JSON files and combine it with the syntax metrics. Save the data as `whygoal_stats.csv`

#### `make_fig3.py`
Use the entire dataset to plot the cumulative time taken for Valid/Invalid/Unknown answers to be returned. Renders `line_graph.pdf` to `paper` folder and prints values for the 99th percentile.

#### `whygoal_test.csv`, `whygoal_valid_test.csv`
Disjoint partitions of `whygoal_stats.csv` for testing (25%) and training/validation (75%) respectively

#### `compare_regressors.py`
Perform KFold cross-validation on the training set to compare a number of regressor implementations from Sci-kit Learn. Renders `compare_regressors.pdf` which is the full version of Table 2 in the paper.

#### `permute_rankings.py`
Find values for the 'Random' strategy (either train or test) by averaging values for all possible rankings. Is slow because it has 8! rankings to get through.

#### `output_eval_files.py`
Outputs several data files used in the Evaluation section:
- `forest.json`: a JSON representation of the trained random forest - suitable for use when compiling the OCaml binary
- `data_for_second_barchart.csv`: results for each prover and strategy for the test goals
- `data_for_second_linegraph.csv`: how long each strategy took to return a Valid/Invalid answer for the test set
- `feature_importances.txt`: These relevance metrics are computed by Sci-kit Learn's Random Forest implementation: they describe the proportion of decisions based on each input variable across all decision trees in Where4's random forest.

#### `barchart2.py`
Renders `barcharts2.pdf` to the `paper` folder. Similar to `make_fig1.py` but reads from `data_for_second_barchart.csv` and includes theoretical strategies and Where4 results (result of choosing the __first__ solver in each ranking).

#### `plot_second_linegraph.py`
The cumulative time taken for the three theoretical strategies and Where
to find an answer to the goals in the test dataset. Uses data stored in `data_for_second_linegraph.csv` - particularly important for the time-consuming 'Random' calculations. Renders `line_graph_eval_provers.pdf` to the `paper` folder. Also prints the average times File/Theory/Goal times used in Table 3.

#### `thresholds.py`
Parameterise Where4's performance by using a threshold, reading data from `data_for_second_linegraph.csv`. Renders `thresholds.pdf` to `paper` folder.
These plots show the effect of the threshold on the time taken for a response (top) and number of goals which can be proved (bottom).
