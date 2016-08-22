# Predicting SMT Performance for Software Verification
## Andrew Healy, Rosemary Monahan, James, F. Power
### Dept. of Computer Science, Maynooth University, Ireland

### [F-IDE 2016](https://sites.google.com/site/fideworkshop2016/), support data

This repository contains data measuring 8 SMT solvers' performance on the
[Why3](http://why3.lri.fr/) examples dataset. We record the result returned by [Alt-Ergo](https://alt-ergo.ocamlpro.com/) (versions 0.95.2 and 1.01), [CVC3](http://www.cs.nyu.edu/acsys/cvc3/), [CVC4](http://cvc4.cs.nyu.edu/web/), [veriT](http://www.verit-solver.org/), [Yices](http://yices.csl.sri.com/), and [Z3](https://github.com/Z3Prover/z3) (versions 4.3.2 and 4.4.1). We also measure the time taken
by the solver to return the result.

- `data` : this folder contains a subfolder for each file in the examples repository. Each folder contains:
 - `<name>.mlw` the WhyML file sent to Why3
 - `<name>.json` a JSON dictionary with the following schema:
   - { <name of file>: { <theories in file> : { <goals in theory> : {   } } } 
