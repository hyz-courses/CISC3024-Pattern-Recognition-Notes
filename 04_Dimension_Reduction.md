# 4.0 A Quick View
### What does it do?
Dimension Reduction:
- Reduces the dimension of data.
	- Changes the data representation into a lower-dimensional one.
- It preserves the structure of the data.
- Usually unsupervised.
### Why do we need DR?
- Computation Complexity
- Pre-processing stage before further learning
- Data Visualization
- Data Interpretation

# 4.1 Singular Value Decomposition (SVD) 奇异值分解
## 4.1.1 Definition
- [i] The Singular Value Decomposition process could be described as follows.
$$
A_{m\times n}=U_{m\times m}S_{m\times n}V_{n\times n}^\top
$$
where,
- $A$ is any $m\times n$ matrix.
- $U$ is any $m\times m$ orthogonal matrix. 正交矩阵
	- $U^\top=U^{-1}$
	- $UU^\top=U^\top U=I$
- $S$ is any $m\times n$ diagonal matrix. 对角矩阵
	- Singular values $\sigma_1>\sigma_2>\cdots>\sigma_{\min(m,n)}>0$ is the main diagnal of $S$.
		- $S=\begin{bmatrix}\sigma_1 & 0 & \cdots & 0 & \cdots & 0 \\ 0 & \sigma_2 & \cdots & 0 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots & \ddots & \vdots\\ 0 & 0 & \cdots & \sigma_m & \cdots & 0\end{bmatrix}$
	- $\sigma_1^2>\sigma_2^2>\cdots>\sigma_{\min(m,n)^2}$ are the **eigenvalues** of $AA^\top$ and $A^\top A$.
- $V$ is any $n\times n$ orthogonal matrix.

## 4.1.2 Calculation Procedures

### Problem Setup
**Given**
- A matrix $A=\begin{bmatrix}2 & 0 & 1 \\ -1 & 2 & 0\end{bmatrix}$.
**Do**
- Find $U$, $S$, and $V$ for Singular Value Decomposition.
### Basic Knowledge
- $AA^\top$
	- $=\Bigl(USV^\top\Bigr)\Bigl(USV^\top\Bigr)^\top$
	- $=\Bigl(USV^\top\Bigr)\Bigl(VS^\top U^\top\Bigr)$
	- $=US\Bigl(V^\top V\Bigr) S^\top U^\top$
	- $=U\Bigl(SS^\top\Bigr)U^\top$

- $A^\top A$
	- $=\Bigl(USV^\top\Bigr)^\top\Bigl(USV^\top\Bigr)$
	- $=\Bigl(VS^\top U^\top\Bigr)\Bigl(USV^\top\Bigr)$
	- $=VS^\top\Bigl(U^\top U\Bigr)SV^\top$
	- $=VS^\top SV^\top$
### Step 1. Calculate $AA^\top$ and $A^\top A$
Known that:
$$
A=\begin{bmatrix}2 & 0 & 1 \\ -1 & 2 & 0\end{bmatrix}, \ A^\top=\begin{bmatrix}2 & -1 \\ 0 & 2 \\ 1 & 0\end{bmatrix}
$$
Therefore, we could get:
$$
AA^\top=\begin{bmatrix}2 & 0 & 1 \\ -1 & 2 & 0\end{bmatrix}\begin{bmatrix}2 & -1 \\ 0 & 2 \\ 1 & 0\end{bmatrix}=\begin{bmatrix}5 & -2 \\ -2 & 5\end{bmatrix}
$$
$$
A^\top A=\begin{bmatrix}2 & -1 \\ 0 & 2 \\ 1 & 0\end{bmatrix}\begin{bmatrix}2 & 0 & 1 \\ -1 & 2 & 0\end{bmatrix}=\begin{bmatrix}5 & -2 & 2 \\ -2 & 4 & 0 \\ 2 & 0 & 1\end{bmatrix}
$$
### Step 2. Eigenvalues and $S$
As we obtained $AA^\top$ and $A^\top A$, we can get their eigenvalues, and construct $S$ matrix.
From the definition of Eigen Values:
$$
AA^\top = \lambda I
$$
where $\lambda$ is the eigenvalue of $AA^\top$.
- $\implies |AA^\top-\lambda I| = 0$
- $\implies \Biggl|\begin{pmatrix}5 & -2 \\ -2 & 5\end{pmatrix}-\lambda\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}\Biggr|=0$
- $\implies \begin{vmatrix}5-\lambda & -2 \\ -2 & 5-\lambda\end{vmatrix}=0$
- $\implies (5-\lambda)^2-4=0$
- $\implies \lambda^2-10\lambda+21=0$
- $\implies (\lambda-3)(\lambda-7)=0$
- $\implies \begin{cases}\lambda_1=7 \\ \lambda_2=3\end{cases}, \ \begin{cases}\sigma_1=\sqrt{\lambda_1}=\sqrt{7}\\ \sigma_2=\sqrt{\lambda_2}=\sqrt{3}\end{cases}$

Therefore, the diagonal matrix $S$ would be:
$$
S=\begin{pmatrix}\sigma_1 & 0 \\ 0 & \sigma_2\end{pmatrix}=\begin{pmatrix}\sqrt{7} & 0 \\ 0 & \sqrt{3}\end{pmatrix}
$$
### Step 3. Find $U$
We need to find $U$ using the eigenvalues we obtained from Step 2. Again, by the property of eigenvalues of a matrix:
$$
\forall x\in \mathbb{R}^m, \ (AA^\top-\lambda I)x=0
$$
For $\lambda_{1} = \sqrt{7}$:
- $(AA^\top-\lambda I)x_1 = 0$
- $\implies \Biggl(\begin{pmatrix}5 & -2 \\ -2 & 5\end{pmatrix}-\sqrt{7}\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}\Biggr)x=0$
- $\implies \begin{pmatrix}5-\sqrt{7} & -2 \\ -2 & 5-\sqrt{7}\end{pmatrix}x=0$
- 
# 4.2 Principle Component Analysis (PCA) 主成分分析
### Problem Setup
**Given**
- An $m\times n$ training data set $X=\begin{pmatrix}x^{(1)} & x^{(2)} & \cdots & x^{(m)}\end{pmatrix}$
	- where $x^{(i)}\in\mathbb{R}^n$
	- that is, $X=\begin{pmatrix}x_{1}^{(1)} & x_{1}^{(2)} & \cdots & x_{1}^{(m)} \\ x_{2}^{(1)} & x_{2}^{(2)} & \cdots & x_{2}^{(m)} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n}^{(1)} & x_{n}^{(2)} & \cdots & x_{n}^{(m)}\end{pmatrix}$
	- Structural Analysis:
		- $n$ is the *dimension* of a data sample. Each row is a feature.
		- $m$ is the *amount* of data sample. Each column is a dataset.

**Do**
- Reduces the dataset from $n$-dimensions to $k$-dimensions.
	- That is, to convert each feature from a $n$-d vector to a $k$-d vector;
	- Namely, to convert $X$ from an $n\times m$ matrix into a $k\times m$ matrix.

## 4.2.1 Data Preprocessing: Mean Normalization
**Given**
- The $m\times n$ training dataset $X=\begin{pmatrix}x^{(1)} & x^{(2)} & \cdots & x^{(m)}\end{pmatrix}$.
**Do**
1. Calculate feature mean for each vectors:
	- $\mu=\begin{pmatrix}\mu_1 \\ \mu_2 \\ \cdots \\ \mu_m\end{pmatrix}$
	- where $\mu_j=\sum_{i=1}^{n}x_{j}^{(i)}$.
	- A mean of a feature with respect to all the data samples.
2. Feature scaling:
	- For each row of $X$, that is a set of a specific feature of each data sample,
		- Reduce each value on this row by the row mean.
		- $\begin{pmatrix}x_j^{(1)}-\mu_j & x_j^{(2)}-\mu_j & \cdots & x_j^{(m)}-\mu_j\end{pmatrix}$
### What it does:
What we eventually get is:
- A scaled version of a dataset.
- Since different features may have their own range of values, which could vary, we need to normalize all features into a unified range of values.
	- E.g.: House Size is around 200 squared meters, while the price could be around 30,000.

## 4.2.2 Compute Covariance Matrix
**Given**
- The normalized version of dataset $X$.
**Do**
1. Compute the covariance matrix by:
$$
\Sigma = \dfrac{1}{m}\sum_{i=1}^m x^{(i)}(x^{(i)})^\top=\dfrac{1}{m}XX^\top
$$
2. Compute eigenvectors using *Singular Value Decomposition*.
$$
U_{n\times n}S_{n\times n}V_{n\times n}=svd(\Sigma)
$$

## 4.2.3 Dimension Reduction on $U$


# 4.3 Linear Discriminant Analysis (LDA)