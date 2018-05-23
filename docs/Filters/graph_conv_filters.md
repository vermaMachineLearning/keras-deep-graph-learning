#### Standard Polynomial

```python
standard_poly(A, poly_degree=3)
```
Computes Standard Polynomial function. Current implementation complexity is $O(N^2)$.


__Inputs__:

* __A__ : 2D Tensor, graph adjacency matrix or (normalized) Laplacian.
* __poly_degree__: Integer, polynomial degree (default=1). 

__Outputs__:

* 3D Tensor, containing standard polynomial powers of graph adjacency matrix or Laplacian.

----

#### Chebyshev Polynomial

```python
chebyshev_poly(A, poly_degree=3)
```
Computes Chebyshev Polynomial function. Current implementation complexity is $O(N^2)$.

__Inputs__:

* __A__ : 2D Tensor, graph adjacency matrix or (normalized) Laplacian.
* __poly_degree__: Integer, polynomial degree (default=1). 

__Outputs__:

* 3D Tensor, containing chebyshev polynomial powers of graph adjacency matrix or Laplacian.

__References__: Defferrard, Michaël, Xavier Bresson, and Pierre Vandergheynst. "Convolutional neural networks on graphs with fast localized spectral filtering." In Advances in Neural Information Processing Systems, pp. 3844-3852. 2016.

----

#### Random Walk Polynomial

```python
chebyshev_poly(A, poly_degree=3)
```
Computes Random Walk Polynomial function. Current implementation complexity is $O(N^2)$.

__Inputs__:

* __A__ : 2D Tensor, graph adjacency matrix or (normalized) Laplacian.
* __poly_degree__: Integer, polynomial degree (default=1). 

__Outputs__:

* 3D Tensor, containing chebyshev polynomial powers of graph adjacency matrix or Laplacian.

----



#### Cayley Polynomial

```python
cayley_poly(A, poly_degree=3)
```
Computes Cayley Polynomial function. Current implementation complexity is $O(N^3)$.

__Inputs__:

* __A__ : 2D Tensor, graph adjacency matrix or (normalized) Laplacian.
* __poly_degree__: Integer, polynomial degree (default=1). 

__Outputs__:

* 3D Tensor, containing cayley polynomial powers of graph adjacency matrix or Laplacian.

__References__: Levie, Ron, Federico Monti, Xavier Bresson, and Michael M. Bronstein. "Cayleynets: Graph convolutional neural networks with complex rational spectral filters." arXiv preprint arXiv:1705.07664 (2017).

----

#### Combine Polynomial

```python
combine_poly(A, B, poly_degree=3)
```
Computes combination of polynomial function.

__Inputs__:

* __A__ : 2D Tensor, graph adjacency or (normalized) Laplacian or cayley matrix.
* __B__ : 2D Tensor, graph adjacency matrix or (normalized) Laplacian or cayley matrix.
* __poly_degree__: Integer, polynomial degree (default=1). 

__Outputs__:

* 3D Tensor, containing combine polynomial powers of graph adjacency  or Laplacian or cayley matrix.



