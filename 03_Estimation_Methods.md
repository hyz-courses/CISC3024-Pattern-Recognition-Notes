# 3.1 $l$-norms and Distance Metrics
## 3.1.1 $l$-norms
$X$ is a column vector in $\mathbb{R}^N$ space.
- $X=\begin{bmatrix}x_1 \\ x_2 \\ \cdots \\ x_N\end{bmatrix}$
### $l_0$-norm
$$
\|X\|_0 \equiv \sum_{i=1}^{N}|x_i|^0 = |x_1|^0 + |x_2|^0 + \cdots + |x_N|^0
$$
- $l_0$-norm is the number of non-zero elements in vector $X$.
	- Defined that $0^0=0$.
- Application:
	- $\|X\|_0$ is very small $\iff$ The vector $X$ is very sparse/shallow. 
	- Minimize $\|X-Y\|_0 \iff$ Minimize the difference between $X$ and $Y$. 

### $l_1$-norm
$$
\|X\|_1 \equiv \sum_{i}^{N}|x_i|=|x_1|+|x_2|+\cdots+|x_N|
$$
- $l_1$-norm is the sum of absolute values of elements in vector $X$.
- $l_1$-norm is also called
	- Taxicab norm
	- Manhattan norm
- Application:
	- Minimize $\|X\|_1 \iff$ Minimize total value of non-zero element sums. Similar results as minimize $\|X\|_0$.

### $l_2$-norm
$$
\|X\|_2 \equiv (\sum_{i=1}^{N}|x_i|^2)^{\dfrac{1}{2}}=\sqrt{x_1^2+x_2^2+\cdots+x_N^2}
$$
- $l_2$-norm can be expressed as matrix format $\|X\|_2\equiv\sqrt{X^\top X}$
- $l_2$-norm is also called
	- Euclidean norm
- Application:
	- Minimize $\|X\|_2 \iff$ Make matrix more sparse.
### $l_{\infty}$-norm
$$
\|X\|_{\infty} \equiv max(|x_1|,|x_2|,\cdots,|x_N|)
$$
- $l_\infty$-norm takes the maximum of absolute values of elements in vector $X$.
- $l_\infty$-norm is also called
	- Maximum norm

### $l_p$-norm
$$
\|X\|_p \equiv \Bigl(\sum_{i=1}^{N}|x_i|^p\Bigl)^{\dfrac{1}{p}}=(|x_1|^2+|x_2|^2+\cdots+|x_N|^p)^{\dfrac{1}{p}}
$$
- $l_p$-norm is a general form of $l$-norm, where $p\geq 0$.
	- $p=0$, $l_0$-norm,
	- $p=1$, $l_1$-norm,
	- $p=2$, $l_2$-norm,
	- ...,
	- $p\rightarrow\infty$, $l_\infty$-norm.

![[Graphical Representation of l-norms.png]]
## 3.1.2 Distance Metrics
### Euclidean Distance
**Given**
- Two datasets
	- $X=\begin{bmatrix}x_1 \\ x_2 \\ \cdots \\ x_N\end{bmatrix}, \ Y=\begin{bmatrix}y_1 \\ y_2 \\ \cdots \\ y_N\end{bmatrix}\in\mathbb{R}^N$
**Do**
- Euclidean Distance:
	- $d_E(X,Y)=\sqrt{\sum_{i=1}^{N}(x_i-y_i)^2}=\sqrt{(X-Y)^\top(X-Y)}$
	- The straight-line distance between X and Y.
	- Also called $L_2$ distance.

### Mahalanobis Distance
**Given**
- An observation $X=\begin{bmatrix}x_1 \\ x_2 \\ \cdots \\ x_N\end{bmatrix}$.
- A set of observations with 
	- mean $\mu=\begin{bmatrix}\mu_1 \\ \mu_2 \\ \cdots \\ \mu_N\end{bmatrix}$
	- Covariance matrix $\Sigma$
**Do**
- Mahalanobis Distance
	- $D_M(X,Y)=\sqrt{(X-\mu)^\top\Sigma^{-1}(X-\mu)}$
	- It is a measure of the distance between
		- a point, and
		- a distribution
	- Reverts to Euclidian distance when $\Sigma=I$.
# 3.2 Parameter Estimation
Recall the Bayes Formula
$$
P(\omega_j|X)=\dfrac{P(X|\omega_j)P(\omega_j)}{p(X)}
$$
- We need to retrieve prior probability $P(\omega_j)$ and posterior probability $P(X|\omega_j)$ from training samples.
	- Collect training samples $\{X_1,X_2,\cdots,X_N\}$ distributed according to the unknown $P(X|\omega_j)$.
	- Assumed that $X_1,X_2,\cdots,X_N$ are i.i.d.
		- Independent and identically distributed.
		- Independent: $X_i$ and $X_j$ does not influence each other.
		- Identicall: $X_i\neq X_j \forall i\neq j$.
We are estimating the hyper parameters of
- The distribution $P(X|\omega_j)$, that is
- $\mu_j, \Sigma_j$
# 3.3 Maximum Likelihood Estimation

# 3.4 Bayesian Estimation
