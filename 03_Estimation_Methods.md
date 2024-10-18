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
All we have initially are the training samples.
- We don't directly "know" the prior & posterior probabilities.

Therefore, we need to retrieve prior probability $P(\omega_j)$ and posterior probability $P(X|\omega_j)$ from training samples.
- Collect training samples $\{X_1,X_2,\cdots,X_N\}$ distributed according to the unknown $P(X|\omega_j)$.
- Assumed that $X_1,X_2,\cdots,X_N$ are **Independent and identically distributed** (i.i.d.).
	- Independent: $X_i$ and $X_j$ does not influence each other.
	- Identical: $X_i\neq X_j, \ \forall i\neq j$.

We are estimating the hyper parameters of the posterior distribution $P(X|\omega_j)$, that is
- $\mu_j, \Sigma_j$, as we assumed a normal distribution.

Our next goal is to estimate $\mu_j$ and $\Sigma_j$. 
- Parametric Form
	- Maximum Likelihood Estimation (MLE)
	- Bayesian Estimation (BE)
- Nonparametric Form
# 3.3 Maximum Likelihood Estimation

## 3.3.1 Find the best $\theta$ : Log-Likelihood.
**Given**
- The set of i.i.d. training Examples:
	- $X=\{x_1,x_2,\cdots,x_N\}$, where 
		- $\forall k=1,2,\cdots,N, \ x_k \sim p(x|\theta)$
- The parameters to be estimated: $\theta$
**Do**
We derive the objective function:
- $p(X|\theta)\equiv p(x_1,x_2,\cdots,x_N|\theta)$
	- $=\prod_{k=1}^{N}p(x_k|\theta)$
- $p(X|\theta)$ is the **Likelihood** of $\theta$ with respect to $X$.

To find a best $\theta$, we derive a maximum likelihood estimation:
$$
\hat{\theta}_{ML}=argmax_\theta p(X|\theta)=argmax_\theta\prod_{k=1}^{N}p(x_k|\theta)
$$
That is, we want to find the $\theta$ that gives the **Maximum Likelihood** on $X$.

### Log-likelihood
For optimization purposes, we derive a log-likelihood function that preserves the monotonicity of the original MLE.
$$
L(\theta)=\ln p(X|\theta)=\sum_{k=1}^{N}\ln p(x_k|\theta)
$$
As the monotonicity is preserved, we would derive that
- $\hat{\theta}_{ML}=argmax_\theta \Bigl[p(X|\theta)\Bigr]$
	- $\iff \hat{\theta}_{ML}=argmax_\theta \Bigl[L(\theta)\Bigr]$
	- $\iff \hat{\theta}_{ML}=argmax_\theta \Bigl[\sum_{k=1}^{N}\ln p(x_k|\theta)\Bigr]$

Equivalently, we find the $\theta$ that gives the maximum $L(\theta)$ now.

To find the $\theta$ that maximizes $L(\theta)$, we find:
$$
\hat{\theta}_{ML}: \ \dfrac{\partial L(\theta)}{\partial\theta}=0
$$
That is,
$$
\hat{\theta}_{ML}: \ \sum_{k=1}^{N}\dfrac{\partial\Bigl[\ln p(x_k|\theta)\Bigr]}{\partial\theta}=0
$$
## 3.3.2 $\mu$ unknown, $\Sigma$ known; $\theta=\{\mu\}$.
### Univariate & Multivariate Case ($x\in \mathbb{R}^{N^+}$)
$$p(x_k|\mu)=\dfrac{1}{\Bigl(2\pi\Bigr)^{\dfrac{d}{2}}|\Sigma|^{\dfrac{1}{2}}}e^{-\dfrac{1}{2}(x_k-\mu)^\top\Sigma^{-1}(x_k-\mu)}$$
- $\implies \ln p(x_k|\mu)=-\dfrac{1}{2}\ln\Bigl[(2\pi)^d|\Sigma|\Bigl]$
	- $-\dfrac{1}{2}(x_k-\mu)^\top\Sigma^{-1}(x_k-\mu)$

- $\implies \ln p(x_k|\mu)=-\dfrac{1}{2}\ln\Bigl[(2\pi)^d|\Sigma|\Bigl]$
	- $-\dfrac{1}{2}(x_k^\top-\mu^\top)\Sigma^{-1}(x_k-\mu)$

- $\implies \ln p(x_k|\mu)=-\dfrac{1}{2}\ln\Bigl[(2\pi)^d|\Sigma|\Bigl]$
	- $-\dfrac{1}{2}(x_k^\top\Sigma^{-1}-\mu^\top\Sigma^{-1})(x_k-\mu)$

- $\implies \ln p(x_k|\mu)=-\dfrac{1}{2}\ln\Bigl[(2\pi)^d|\Sigma|\Bigl]$ (Constant term since $\Sigma$ known)
	- $-\dfrac{1}{2}x_k^\top\Sigma^{-1}x_k$ (Constant term since $x_k$ is pre-defined)
	- $-\dfrac{1}{2}x_k^\top\Sigma^{-1}\mu$
	- $+\dfrac{1}{2}\mu^\top\Sigma^{-1}x_k$
	- $-\dfrac{1}{2}\mu^\top\Sigma^{-1}\mu$

- [?] $\implies\dfrac{\partial\ln p(x_k|\mu)}{\partial\mu}=\Sigma^{-1}(x_k-\mu)$ 

As we required
$$
\hat{\mu}_{ML}: \ \dfrac{\partial L(\mu)}{\partial\mu} =  \ \sum_{k=1}^{N}\dfrac{\partial\Bigl[\ln p(x_k|\mu)\Bigr]}{\partial\mu}=0
$$
- $\implies \sum_{k=1}^{N}\Sigma^{-1}(x_k-\hat{\mu})=0$

- $\implies \Sigma^{-1}\sum_{k=1}^{N}(x_k-\hat{\mu})=0$

- $\implies\sum_{k=1}^{N}(x_k-\hat{\mu})=0$

- $\implies \Bigl[\sum_{k=1}^{N}x_k\Bigr]-N\hat{\mu}=0$

- [*] $\implies \hat{\mu}=\dfrac{1}{N}\sum_{k=1}^{N}x_k$

## 3.3.3 $\mu$ unknown, $\Sigma$ unknown; $\theta=\{\mu,\Sigma\}$
### Univariate Case ($x\in\mathbb{R}$)
$$
p(x_k|\theta)=\dfrac{1}{\sigma\sqrt{2\pi}}e^{-\dfrac{(x_k-\mu)^2}{2\sigma^2}}
$$
where $\theta=\begin{bmatrix}\mu \\ \sigma^2\end{bmatrix}=\begin{bmatrix}\theta_1 \\ \theta_2\end{bmatrix}$

- $\implies \ln p(x_k|\theta)=-\dfrac{1}{2}\ln(2\pi\sigma^2)-\dfrac{1}{2\sigma^2}(x_k-\mu)^2$
	- $=-\dfrac{1}{2}\ln(2\pi\theta_2)-\dfrac{1}{2\theta_2}(x_k-\theta_1)^2$

- $\implies \dfrac{\partial\ln p(x_k|\theta)}{\partial\theta}=\begin{bmatrix}\dfrac{x_k-\theta_1}{\theta_2}\\-\dfrac{1}{2\theta_2}+\dfrac{(x_k-\theta_1)^2}{2\theta_2^2}\end{bmatrix}$

Again, to find the $\theta$ that minimizes the MLE, we let
$$
\dfrac{\partial\ln L(\theta)}{\partial\theta}=\sum_{k=1}^{N}\dfrac{\partial\ln p(x_k|\theta)}{\partial\theta}=0
$$
- $\implies \begin{bmatrix}\sum_{k=1}^{N}\dfrac{x_k-\theta_1}{\theta_2}\\\sum_{k=1}^{N}\Bigl(-\dfrac{1}{2\theta_2}+\dfrac{(x_k-\theta_1)^2}{2\theta_2^2}\Bigr)\end{bmatrix}=0$

Namely,
- $\sum_{k=1}^{N}\dfrac{x_k-\hat{\theta_1}}{\hat{\theta_2}}=0$
	- $\implies \dfrac{1}{\hat{\theta_2}}\sum_{k=1}^{N}(x_k-\hat{\theta_1})=0$
	- $\implies \sum_{k=1}^{N}(x_k-\hat{\theta_1})=0$
	- $\implies\Bigl(\sum_{k=1}^{N}x_k\Bigr)-N\hat{\theta_1}=0$
	- [*] $\implies \hat{\mu}=\hat{\theta_1}=\dfrac{1}{N}\sum_{k=1}^{N}x_k$

- $\sum_{k=1}^{N}\Bigl(-\dfrac{1}{2\hat{\theta_2}}+\dfrac{(x_k-\hat{\theta_1})^2}{2\hat{\theta_2}^2}\Bigr)=0$
	- $\implies \dfrac{\sum_{k=1}^{N}(x_k-\hat{\theta_1})^2}{2\hat{\theta_2}^2}=\dfrac{N}{2\hat{\theta_2}}$
	- $\implies N\hat{\theta_2}=\sum_{k=1}^{N}(x_k-\hat{\theta_1})^2$
	- $\implies \hat{\theta_2}=\dfrac{1}{N}\sum_{k=1}^{N}(x_k-\hat{\theta_1})^2$
	- [*] $\implies \hat{\sigma}^2=\hat{\theta_2}=\dfrac{1}{N}\sum_{k=1}^{N}(x_k-\hat{\mu})^2$

### Multivariate Case ($x\in\mathbb{R}^D,D>1$)
$$
p(x_k|\theta)=
\dfrac{1}{\Bigl(2\pi\Bigr)^{\dfrac{d}{2}}|\Sigma|^{\dfrac{1}{2}}}e^{-\dfrac{1}{2}(x_k-\mu)^\top\Sigma^{-1}(x_k-\mu)}
$$
where $\theta=\begin{bmatrix}\mu \\ \Sigma\end{bmatrix}=\begin{bmatrix}\theta_1 \\ \theta_2\end{bmatrix}$.

Similarly, we have
- [*] $\hat{\mu}=\dfrac{1}{N}\sum_{k=1}^{N}x_k$  
- [*] $\hat{\Sigma}=\dfrac{1}{N}\sum_{k=1}^{N}(x_k-\hat{\mu})(x_k-\hat{\mu})^\top$ 

# 3.4 Bayesian Estimation
## 3.4.0 Difference between BE and MLE
- In ML estimation, $\theta$ was considered a **parameter** with a fixed value.
- In Bayesian estimation however, $\theta$ is considered an **unknown random vector**.
	- which is described by a P.D.F $p(\theta)$.

## 3.4.1 Find the best $\theta$ : Bayes Formula
**Given**
- The set of i.i.d. training examples:
	- $X=\{x_1,x_2,\cdots,x_N\}$, where
		- $\forall k=1,2,\cdots,N, \ x_k\sim p(x|\theta)$.
- The parameters to be estimated: $\theta$.
**Do**
As $\theta$ is regarded to be random, we compute the maximum of $p(\theta|X)$.

From Bayes formula, 
$$
p(\theta|X)=\dfrac{p(X|\theta)p(\theta)}{p(X)}
$$
We find the  with the best Maximum Aposterior Probability.
$$
\hat{\theta}_{MAP}=argmax_{\theta}\Bigl[p(\theta|X)\Bigr]
$$
Namely, 
- $\hat{\theta}_{MAP}=argmax_\theta\Bigl[p(X|\theta)P(\theta)\Bigr]$
- $\implies \hat{\theta}_{MAP}: \  \dfrac{\partial\Bigl[p(X|\theta)p(\theta)\Bigr]}{\partial\theta}=0$
- $$