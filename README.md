# Non-extensive sparse factorization of the H2 matrix.

This is a Python implementation of the [non-extensive sparse factorization of the H2 matrix](https://arxiv.org/abs/1705.04601). The algorithm takes as input N by N H2 matrix A and returns the factorization A = USV, where S is an N by N sparse matrix, U and V are N by N orthogonal matrices that are products of
block-diagonal and permutation matrices. The factorization allows using sparse solvers for the solution of the systems with the H2 matrices.


## Requirements

The minimum requirement by CE is [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [matplotlib](http://matplotlib.org/),  [h2tools](https://bitbucket.org/muxas/h2tools).

## Quick Start

You can find Jupyter notebook with examples in folder "examples".
