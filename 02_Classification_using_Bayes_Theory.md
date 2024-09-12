# 2.1 Bayes Decision Theory
Basic Assumptions
- The decision problem is posed in probabilistic terms.
- **ALL** relevant probability values are known.

## 2.1.1 Process
- **Given:**
	1. A test sample $x$.
		- Contains features $x=[x_1,x_2,...x_l]^T$.
		- Often reduced, removed some non-discriminative (un-useful) features.
	2. A list of classes/patterns $\omega=\{\omega_1,\omega_2,...\omega_c\}$.
		- Defined by human-being.
	3. A classification method $M$.
		- A **database** storing multiple samples with the same type of $x$.
		- Each sample is assigned to an arbitrary class $\omega_{any}\in\{\omega_1,\omega_2,...\omega_c\}$.
- **Do:**
	- $\{P(\omega_1|x),...,P(\omega_c|x)\}\leftarrow classify(M,x,\omega)$
	- That is, for all the possible classes, find:
		- The probability that the given $x$ belongs to that class.
- **Get:**
	-  $\omega_{target}(x)=argmax_i[P(\omega_i|x)], i\in[1,c]$.
	- That is, assign $x$ a class/pattern from $\omega$ with the **most probable** one.

**Example**
MNIST database.
- Test sample: 
	- $x=$ A $28 \times 28$ grayscale image of a hand-written number. 
- Set of classes: 
	- $\omega$ = $\{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$.
- Classification Method: 
	- Derived from 10,000 of $28\times28$ similar gray-scale images.
- Process:
	- Given an image, using the classification method, get a list of probabilities $P(\omega) = \{P(\omega_1),P(\omega_2),...,P(\omega_c)\}$.
	- Select the $\omega_i$ with the largest probability $P(\omega_i)$, that is $selected=argmax[P(\omega_i)]$.

## 2.1.2 Properties of Variables.
- The set of all classes $\omega$ :
	- $c$ available classes: $\omega = \{\omega_1, \omega_2, ..., \omega_c\}$
- Prior Probabilities $P(\omega)=\{P(\omega_1),P(\omega_2),...,P(\omega_c)\}$ :
	- Probability Distribution of random variable $\omega_j$ in the database.
		- The fraction of samples in the database that belongs to class $\omega_j$.
		- $P(\omega)$ is the prior knowledge on $\omega=\{\omega_1,\omega_2,...,\omega_c\}$.
	- It is Non-Negative.
		- $\forall i \in [1,c], P(\omega_i) \geq 0$.
		- The probabilities of all classes are greater-or-equal to 0.
	- It is Normalized.
		- $\sum_{i=1}^{c}P(\omega_i)=1$.
		- The sum of the prior probabilities of all classes is $1$.

# 2.2 Prior & Posterior Probabilities 先验与后验概率
## 2.2.1 [DEF] Prior Probability 先验概率
- Decision **BEFORE** Observation (Naïve Decision Rule). 
	- Don't care about test sample $x$.
	- Given $x$, always choose the class that:
		- has the most member in the database.
		- i.e., has the highest prior probability.
- Classification Process:
	1. $\omega=\{\omega_1,\omega_2,...,\omega_c\}$.
	2. By counting the number of members $Num(\omega_i)$ for each class $\omega_i\in\omega,i\in[1,c]$, we get the prior probabilities $P(\omega) = \{P(\omega_1),P(\omega_2),...,P(\omega_c)\}$.
	3. Then, classify $x$ directly into $argmax_i[P(\omega_i)]$.
- The decision is the same all the time obviously, and the prob. of a right guess is $\dfrac{1}{c}$. 

## 2.2.2 [DEF] Posterior Probability 后验概率
- Decision **WITH** Observation. 
	- Cares about test sample $x$.
	- Considering $x$, as well as the prior probabilities $P(\omega) = \{P(\omega_1),P(\omega_2),...,P(\omega_c)\}$, 
		- and give $x$ the class with the biggest posterior probability.
- **Posterior Probability:**
	- [DEF] Posterior Probability of a class $\omega_j$ on test sample $x$:
		- Given test sample $x$, how possible does $x$ could be classified into class $\omega_j$.
	- $P(\omega_j|x)=\dfrac{p(x|\omega_j)P(\omega_j)}{p(x)}$, $Posterior=\dfrac{Likelihood\times Prior}{Evidence}$.
		- $p(x|\omega_j)$: **Likelihood (KNOWN)** 
			- The fraction of samples stored in the database that
				- is same to $x$, and
				- belongs to class $\omega_j$.
		- $P(\omega_j)$: **Prior probability of class $\omega_j$ (KNOWN)** 
			- The fraction of samples stored in the database that
				- belongs to class $\omega_j$.
		- $p(x)$: **Evidence (IRRELEVANT)**
			- Unconditional density of $x$. 
			- That is, $p(x)=\sum_{j=1}^{c}p(x|\omega_j)P(\omega_j)$.
- **Special Cases:**
	1. Equal Prior Probability.
		-  $P(\omega_1)=P(\omega_2)=...=P(\omega_c)=\dfrac{1}{c}$.
		- The amount of members in each class are same.
		- Here, posterior probs. $\forall j\in[1,c], P(\omega_j|x)$ is dependent on the likelihoods $P(x|\omega_j)$ only.
	 2. Equal Likelihood.
		 - $P(x|\omega_1)=P(x|\omega_2)=...=P(x|\omega_c)$.
		 - The amount of members that's same to $x$ in each class are the same.
		 - Here, posterior probs. $\forall j\in[1,c], P(\omega_j|x)$ is dependent on the prior probabilities $P(\omega_j)$ only. 
		 - Back to Naïve Decision Rule.

## 2.2.3 Classification Examples
**Given:**
1. Test sample $x\in\{+,-\}$.
2. A list of classes $\omega=\{\omega_1={cancer},\omega_2={no\_cancer}\}$.
3. Classification Method $M$, with known probabilities:
	- Prior Probabilities:
		- $P(\omega_1)=0.008$
		- $P(\omega_2)=1-P(\omega_1)=0.992$
	- Likelihoods:
		- For class $\omega_1={cancer}$:       $P(+|\omega_1)=0.98$, $P(-|\omega_1)=0.02$
		- For class $\omega_2={no\_cancer}$: $P(+|\omega_2)=0.03$, $P(-|\omega_2)=0.97$. 
**Classification:**
- Given a test sample $x=+$.
	- The prob. that this person gets cancer is:
		- $P(\omega_1|+)=\dfrac{P(+|\omega_1)\times P(\omega_1)}{P(+)}=\dfrac{0.98\times0.008}{P(+)}=\dfrac{0.00784}{P(+)}$.
	- The prob. that this person doesn't gets cancer is:
		- $P(\omega_2|+) = \dfrac{P(+|\omega_2)\times P(\omega_2)}{P(+)}=\dfrac{0.03\times0.992}{P(+)}=\dfrac{0.02976}{P(+)}$
	- Therefore, the classification result would be:
		- $\omega_{target}=argmax_i[P(\omega_i|+)]$
		  $=argmax_i[\dfrac{P(+|\omega_i)\times P(\omega_i)}{P(x)}]$
		  $=argmax_i[P(+|\omega_i)\times P(\omega_i)]$
		  $=\omega_2$, for $0.00784 < 0.02976$
		- That is, $no\_cancer$.

# 2.3 Loss Functions 决策成本函数
## 2.3.0 Why do we use loss functions?
- Different selection errors may have differently significant consequences, i.e., "losses" or "costs". 不同决策的成本、后果不同。
	- In pure Naïve Bayes classification, we only consider probability.
	- However, 
		- we can tolerate "non-cancer" being classified into "cancer", 
		- while it's more lossy to classify "cancer" into "non-cancer".
	- There is a need to consider this kind of "loss" into our decision method.
- We want to know if the Bayes decision rule is optimal.
	- Need a evaluation method
	- calc how many error you make, sum together
## 2.3.1 Probability of Error
For only two classes:
- If $P(\omega_1|x)>P(\omega_2|x)$, $x\leftarrow\omega_1$. Prob. of error: $P(\omega_2|x)$.
-  If $P(\omega_1|x)<P(\omega_2|x)$, $x\leftarrow\omega_2$. Prob. of error: $P(\omega_1|x)$.
## 2.3.2 Loss Function (i.e., "Cost Function")
**Problem**
- Take action $\alpha_i$ for a given $x$.
	- The action $\alpha_i$: To assign the test pattern $x$ the class $\omega_i$.
- Introduce the loss/cost $\lambda(\alpha_i|\omega_j)$, for the true class $\omega_j$ and action $\alpha_i$ on $x$. 
	- That is, $\lambda(\alpha_i|\omega_j)$ is the cost of classifying **any** sample into class $\omega_i$ when the true class of that sample is $\omega_j$.
	- For instance, $\lambda(\alpha_{cancer}|\omega_{no\_cancer})$ is the cost of diagnosing a patient that actually doesn't have cancer as "having cancer". 
		- (Which by intuition is not as serious as its reverse, therefore the value of this $\lambda$ should also be lower than its reverse.)
- We don't actually know the true class $\omega_j$ for a random sample $x$, so we use the Expected Loss.
	- That is, we consider the "average loss" of classifying $x$ into $\omega_i$ by considering:
		- The loss of classifying $x$ into $\omega_j$ for all $\omega_j \in \omega$.
		- The probability that $x\in\omega_j$, i.e., $P(\omega_j|x)$. 

**[DEF]Expected Loss (Average Loss, Conditional Risk) 期望成本:**
- The expected loss of classifying $x$ into $\omega_i$.
- $R(\alpha_i|x)=\sum_{j=1}^{c}{\lambda(\alpha_i|\omega_j)\times P(\omega_j|x)}$ , where
	- $\lambda(\alpha_i|\omega_j)$: The cost of classifying $x$ into $\omega_i$ under the true class $\omega_j$.
	- $P(\omega_j|x)$: The posterior probability that $x$ belongs to class $\omega_j$.
		- Computed during the Naïve Bayes Classification with $P(\omega_j)$ and $P(x|\omega_j)$. 

**[DEF]Bayes Risk 贝叶斯风险**:
- The modified measurement of the original Bayes Rule.
	- Consider the importance of each error.
	- Consider minimum loss, instead of maximum probability.
- Bayes Risk finds the action that gives the minimum expected loss of $x$.
	- $\alpha(x)=argmin_{\alpha_i\in A}R(\alpha_i|x)$
		- $=argmin_{\alpha_i\in A}\sum_{j=1}^{c}\lambda(\alpha_i|\omega_j)P(\omega_j|x)$

**Derivation: For a 2-class problem**
- Known:
	- Test sample $x$.
	- Classes $\omega = \{\omega_1,\omega_2\}$.
	- The calculated posterior probabilities:
		- $P(\omega_1|x)$, $P(\omega_2|x)$.
	- Loss Matrix:$\begin{bmatrix}  \lambda_{11} & \lambda_{12} \\  \lambda_{21} & \lambda_{22}  \end{bmatrix}$, where $\lambda_{ij}=\lambda(\alpha_i|\omega_j)$.
		- $\lambda_{ij}$: The cost of classifying $x$ into $\omega_i$ when the true class of $x$ is $\omega_j$. 
- $\omega_{target}=argmin_{\alpha_i\in A}R(\alpha_i|x)$
- If we choose $\omega_1$, we have:
	- $R(\alpha_1|x)<R(\alpha_2|x)$
	- $\iff \lambda_{11}P(\omega_1|x)+\lambda_{12}P(\omega_2|x)<\lambda_{21}P(\omega_1|x)+\lambda_{22}P(\omega_2|x)$
	- $\iff (\lambda_{21}-\lambda_{11})P(\omega_1|x)>(\lambda_{12}-\lambda_{22})P(\omega_2|x)$
	- $\iff \dfrac{P(\omega_1|x)}{P(\omega_2|x)}>\dfrac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}}$
	- $\iff \dfrac{P(x|\omega_1)P(\omega_1)}{P(x|\omega_2)P(\omega_2)}>\dfrac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}}$
	- $\iff \dfrac{P(x|\omega_1)}{P(x|\omega_2)}>\dfrac{(\lambda_{12}-\lambda_{22})P(\omega_2)}{(\lambda_{21}-\lambda_{11})P(\omega_1)}$
	- $\iff \dfrac{P(x|\omega_1)}{P(x|\omega_2)}>\theta_t$
## 2.3.3 Examples
### Minimum Prob. Error and Minimum Risk

Remark: Gaussian Distribution
- $GD(x)=\dfrac{1}{\sigma\sqrt{2\pi}}e^{-\dfrac{(x-\mu)^2}{2\sigma^2}}$

Given:
- Two probability distributions of evidence $P(x|\omega_j)$ regarding $j\in\{1,2\}$.
	- $P(x|\omega_1)=\dfrac{1}{\sqrt{\pi}}e^{-x^2}$, where $\mu=0, \sigma=\dfrac{1}{\sqrt{2}}$.
	- $P(x|\omega_2)=\dfrac{1}{\sqrt{\pi}}e^{-(x-1)^2}$, where $\mu=1, \sigma=\dfrac{1}{\sqrt{2}}$.
- Loss matrix:
	- $\begin{bmatrix} \lambda_{11} & \lambda_{12} \\ \lambda_{21} & \lambda_{22} \end{bmatrix}= \begin{bmatrix} 0 & 1.0 \\ 0.5 & 0 \end{bmatrix}$
Do:
- The threshold $x_0$ for minimum $P_e$.
	- $P(x_0|\omega_1)=P(x_0|\omega_2)$
		- $\implies\dfrac{1}{\sqrt{\pi}}e^{-x_0^2}=\dfrac{1}{\sqrt{\pi}}e^{-(x_0-1)^2}$
		- $\implies x_0=-x_0+1$,  omitting $x_0=x_0-1$ which is impossible;
		- $\implies x_0=\dfrac{1}{2}$
- The threshold $\hat{x_0}$ for minimum $R(\alpha_i|x)$.
	- $R(\alpha_1|x)=R(\alpha_2|x)$
		- $\implies \dfrac{P(\hat{x_0}|\omega_1)}{P(\hat{x_0}|\omega_2)}=\dfrac{(\lambda_{12}-\lambda_{22})P(\omega_2)}{(\lambda_{21}-\lambda_{11})P(\omega_1)}$
		- $\implies \dfrac{P(\hat{x_0}|\omega_1)}{P(\hat{x_0}|\omega_2)}=\dfrac{(1-0)\times\dfrac{1}{2}}{(0.5-0)\times\dfrac{1}{2}}$
		- $\implies P(\hat{x_0}|\omega_1)=2P(\hat{x_0}|\omega_2)$
		- $\implies\dfrac{1}{\sqrt{\pi}}e^{\hat{-x_0}^2}=2\dfrac{1}{\sqrt{\pi}}e^{-(\hat{x_0}-1)^2}$
		- $\implies-\hat{x_0}^2=\ln2-\hat{x_0}^2+2\hat{x_0}-1$
		- $\implies \hat{x_0}=\dfrac{1-\ln2}{2}<\dfrac{1}{2}$

![[Minimum Prob Error and Minimum Risk.png]]
#### Minimum Error Rate Classification
- A zero-one loss function
	- $\begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix}$
	- All errors are equally costly.
- Conditional Risk:
	- $R(\alpha_i|x)=\sum_{j=1}^{c}\lambda(\alpha_i|x)P(\omega_j|x)$
	- $=\lambda(\alpha_i|\omega_i)P(\omega_i|x)+\sum_{j\neq i}\lambda(\alpha_i|\omega_j)P(\omega_j|x)$
	- $=0+\sum_{j\neq i}1\times P(\omega_j|x)$
	- $=\sum_{j\neq i}P(\omega_j|x)$
	- $=1-P(\omega_i|x)$
# 2.4 Discriminant Functions 判别函数
**[DEF] Discriminant Function**
- If a function $f$ satisfies:
	- If $f(\cdot)$ monotonically increases, and
	- $\forall i\neq j, f(P(\omega_i|x))>f(P(\omega_j|x))$, then
	- $x\rightarrow\omega_i$
- Then, $g_i(x)=f(P(\omega_i|x))$ is a discriminant function.
- That is, this function is able to "tell" a certain one $\omega_i$ from others on any input $x$. 给定一个测试样本$x$，判别函数能够从所有其它分类中挑选一个最可能的$\omega_j$。
	- i.e., it separates $\omega_i$ and $\neg \omega_i$.

 **[PROP] Discriminant Function**
- One function per class.
	- A discriminant function is able to "tell" a certain one $\omega_i$ specifically for any input $x$.
- Various discriminant functions $\rightarrow$ Identical classification results. 样式各异，结果相同。
	- It is correct to say, the discriminant functions:
		- **Preserves** the original monotonical-increase of its inputs.
		- But only changes the changing rate by **processing** the inputs.
	- i.e.,
		- "$\forall i\neq j, f(g_i(x))>f(g_j(x))\land f\nearrow$ "and "$\forall i\neq j, g_i(x)>g_j(x)$" are equivalent in decision.
		- Changing growth rate of input:
			- $f(g_i(x))=k\cdot g_i(x)$, a linear change.
			- $f(g_i(x))=\ln g_i(x)$, a log change, i.e., it grows, but slower as it proceed.
		- Therefore, the discriminant function may vary, but the output is always the same.
- Examples of discriminant functions:
	- Minimum Risk: $g_i(x)=-R(\alpha_i|x) = -\lambda(\alpha_i|x)\times P(\omega_i|x)$, for $i\in[1,c]$
	- Minimum Error Rate: $g_i(x)=P(\omega_i|x)$, for $i\in[1,c]$

**[DEF] Decision Region 决策区域**
- $c$ discriminant functions $\implies$ $c$ decision regions
	- $g_i(x)\implies R_i\subset R^d,i\in[1,c]$
- One function per decision region that is distinct and mutual-exclusive.
	- $R_i=\{x|x\in R^d: \forall i\neq j, g_i(x)>g_j(x)\}$, where
	- $\forall i\neq j, R_i\cap R_j=\emptyset$, and $\cap_{i=1}^{c}R_i=R^d$

**[DEF] Decision Boundaries 决策边界**
- "Surface" in feature space, where ties occur among 2 or more largest discriminant functions.

![[Discriminant Functions.png]]
# 2.5 Bayesian Classification for Normal Distributions
## 2.5.1 Multi-Dimensional Normal Distribution 高维正态分布

**1-D Case**
- $x\sim N(\mu,\sigma):$   $P(x)=\dfrac{1}{\sigma\sqrt{2\pi}}e^{-\dfrac{(x-\mu)^2}{2\sigma^2}}$, where
	- $\mu$ is the mean value.
		- $\mu = E[x]$
	- $\sigma^2$ is the variance.
		- $\sigma = E[(x-\mu)^2]$

**Multivariate Case**
- $X\sim N(\mu,\Sigma):$   $P(X)=\dfrac{1}{|\Sigma|^{\dfrac{1}{2}}\times (2\pi)^{\dfrac{d}{2}}}e^{-\dfrac{1}{2}(X-\mu)^T \Sigma^{-1} (X-\mu)}$
	- Regular Variables:
		- $d$-dimensional random variables: $X=\begin{bmatrix}x_1\\x_2\\...\\x_d\end{bmatrix}$
		- $d$-dimensional mean vector: $\mu=\begin{bmatrix}\mu_1\\\mu_2\\...\\\mu_d\end{bmatrix}=\begin{bmatrix}E(x_1)\\E(x_2)\\...\\E(x_d)\end{bmatrix}$
		- $d\times d$ covariance matrix: $\Sigma = \begin{pmatrix} \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1d} \\ \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2d} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{d1} & \sigma_{d2} & \cdots & \sigma_{dd} \end{pmatrix}=E[(X-\mu)(X-\mu)^T]$
	- Explanations on $-\dfrac{1}{2}(X-\mu)^{T}\Sigma^{-1}(X-\mu)$
		- Parts:
			- $(X-\mu)^T=\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\\...\\x_d-\mu_d\end{bmatrix}^T=\begin{bmatrix}(x_1-\mu_1) & (x_2-\mu_2) & \cdots & (x_d-\mu_d)\end{bmatrix}$
			- $\Sigma^{-1} = \begin{pmatrix} \sigma_{11}' & \sigma_{12}' & \cdots & \sigma_{1d}' \\ \sigma_{21}' & \sigma_{22}' & \cdots & \sigma_{2d}' \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{d1}' & \sigma_{d2}' & \cdots & \sigma_{dd}' \end{pmatrix}$
			- $(X-\mu)=\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\\...\\x_d-\mu_d\end{bmatrix}$
		- Whole:
			- $-\dfrac{1}{2}(X-\mu)^{T}\Sigma^{-1}(X-\mu)$
			- $=-\dfrac{1}{2}\begin{bmatrix}(x_1-\mu_1) & (x_2-\mu_2) & \cdots & (x_d-\mu_d)\end{bmatrix}\begin{pmatrix} \sigma_{11}' & \sigma_{12}' & \cdots & \sigma_{1d}' \\ \sigma_{21}' & \sigma_{22}' & \cdots & \sigma_{2d}' \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{d1}' & \sigma_{d2}' & \cdots & \sigma_{dd}' \end{pmatrix}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\\...\\x_d-\mu_d\end{bmatrix}$
			- $=-\dfrac{1}{2}\begin{bmatrix}a_1 & a_2 & \cdots a_d\end{bmatrix}\begin{bmatrix}x_1-\mu_1\\x_2-\mu_2\\...\\x_d-\mu_d\end{bmatrix}$
			- $=y\geq 0$

**Example: 2-D Case**
- $X\sim N(\mu,\Sigma):$   $P(X)=\dfrac{1}{|\Sigma|^{\dfrac{1}{2}}\times (2\pi)}e^{-\dfrac{1}{2}\begin{bmatrix}(x_1-\mu_1) & (x_2-\mu_2)\end{bmatrix} \Sigma^{-1} \begin{bmatrix}(x_1-\mu_1) \\ (x_2-\mu_2)\end{bmatrix}}$
	- $2$ - dimensional random variable $X$: $X=\begin{bmatrix}x_1\\x_2\end{bmatrix}$
	- $2$ - dimensional mean vector $\mu$: $\mu=\begin{bmatrix}\mu_1\\\mu_2\end{bmatrix}=\begin{bmatrix}E[x_1]\\E[x_2]\end{bmatrix}$
	- $2\times2$ covariant matrix $\Sigma$:
		- $\Sigma=E[(X-\mu)(X-\mu)^T]$
		- $=E(\begin{bmatrix}x_1-\mu_1 \\ x_2-\mu_2\end{bmatrix}\begin{bmatrix}x_1-\mu_1 & x_2-\mu_2\end{bmatrix})$
		- $=\begin{bmatrix}(x_1-\mu_1)^2 & (x_1-\mu_1)(x_2-\mu_2) \\ (x_2-\mu_2)(x_1-\mu_1) & (x_2-\mu_2)^2 \end{bmatrix}$
		- $=\begin{bmatrix}\sigma_1^2 & \sigma \\ \sigma & \sigma_2^2\end{bmatrix}$
- Minimum-error-rate classification:
	- Discriminant Function: $g_i(x)=P(\omega_i|x), i\in[1,c]$
	- Let: $g_i(x)=ln[P(\omega_i|x)]$
		- $\implies g_i(x)=ln[P(X|\omega_i)P(\omega_i)]$
		- $\implies g_i(x)=ln[P(X|\omega_i)]+ln[P(\omega_i)]$
		- 