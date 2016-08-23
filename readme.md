# Predicting SMT Performance for Software Verification
## Andrew Healy, Rosemary Monahan, James, F. Power
### [Principles of Programming](http://www.cs.nuim.ie/research/pop/) Research Group, Dept. of Computer Science, Maynooth University, Ireland

### [F-IDE 2016](https://sites.google.com/site/fideworkshop2016/) support data

This repository contains data measuring 8 SMT solvers' performance on the
[Why3](http://why3.lri.fr/) examples dataset. We record the result returned by [Alt-Ergo](https://alt-ergo.ocamlpro.com/) (versions 0.95.2 and 1.01), [CVC3](http://www.cs.nyu.edu/acsys/cvc3/), [CVC4](http://cvc4.cs.nyu.edu/web/), [veriT](http://www.verit-solver.org/), [Yices](http://yices.csl.sri.com/), and [Z3](https://github.com/Z3Prover/z3) (versions 4.3.2 and 4.4.1). We also measure the time taken
by the solver to return the result.

Python libraries we use: [Pandas](), [Numpy](), [Sci-kit Learn](), [Matplotlib](). All Python files can be run on the command line in the usual way: eg `python <filename.py>`

#### `paper/`
Folder containing latex source files and images for the paper itself

#### `data/`
This folder contains a subfolder for each file in the examples repository. Each folder contains:
 - `<name>.mlw` the WhyML file sent to Why3
 - `<name>.json` a JSON dictionary containing timings and results for various timeout values
 - `stats.json` the syntacic features statically extracted from `<name>.mlw`  (used as independent variables for prediction)

#### `collect_data_fig1_table1.py`
Python script to collect data from the JSON files. Results printed for Table 1 and saved to `fig1_data.csv` to be read in by `make_fig1.py`

#### `make_fig1.py`
Make the first figure (stacked barcharts - 60 second timeout). Uses `fig1_data.csv`. Renders `barcharts.pdf` to `paper` folder

#### `create_stats_df.py`
Collect data from the JSON files and combine it with the syntax metrics. Save the data as `whygoal_stats.csv`

#### `whygoal_test.csv`, `whygoal_valid_test.csv`
Disjoint partitions of `whygoal_stats.csv` for testing (25%) and training/validation (75%) respectively

#### `make_fig3.py`
Use the entire dataset to plot the cumulative time taken for Valid/Invalid/Unknown answers to be returned. Renders `line_graph.pdf` to `paper` folder and prints values for the 99th percentile.
