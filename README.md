# Neural Networks Algorithms

## Hopfield network synchronous update

Formula for x:

x<sub>i</sub> (t+1) = $\sum_{j=1}^n$w<sub>ij</sub>x<sub>j</sub> (t) - b<sub>i</sub>

### How to determine if net is going to converge?

* First of all the matrix needs to be **symmetric**. Thus it needs to be a square matrix satisfying the condition that
**w<sub>ij</sub> = w<sub>ji</sub>** for all i,j indexes.

* Second of all all the values on the diagonal need to be non-negative real numbers. (i.e. **w<sub>ii</sub> $\ge$ 0**)

* Lastly the considered weights matrix must be a **positive definite matrix**. This last condition is satisfied when
[Sylvester's criterion](https://en.wikipedia.org/wiki/Sylvester%27s_criterion) is satisfied.

In order to determine whether the matrix considered is a **positive definite matrix** we can use
**Cholesky decomposition** of the matrix. This method is definitely more **effective** than calculating
all of the **leading principal minors**.

The **Cholesky decomposition** of a square, symmetric positive definite matrix is given as follows:

A = LL<sup>T</sup>, where

A - square, symmetric, positive definite matrix
L - lower triangular matrix
L<sup>T</sup> - L transposed (upper triangular matrix)

In python we can use the **numpy.linalg.cholesky** to check if matrix is a positive definite matrix, because this
will raise an ```LinAlgError``` if the matrix is not positive definite.