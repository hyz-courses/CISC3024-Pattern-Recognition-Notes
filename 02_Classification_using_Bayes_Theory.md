# 2.1 Bayes Decision Theory
Basic Assumptions
- The decision problem is posed in probablistic terms.
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

# 2.2 Prior & Posterior Probabilities
## 2.2.1 [DEF] Prior Probability
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

## 2.2.2 [DEF] Posterior Probability
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

# 2.3 Loss Functions
## 2.3.0 Basics
- Different selections may have different importance.
	- In pure Naive Bayes, we only consider probability.
	- However, 
		- we can tolerate "non-cancer" being classified into "cancer", 
		- while it's more lossy to classify "cancer" into "non-cancer".
	- Need to consider this kind of "loss" into our decision method.
- We want to know if the Bayes decision rule is optimal.
	- Need a evaluation method
	- calc how many error you make, sum together
## 2.3.1 Probability of Error
For only two classes:
- If $P(\omega_1|x)>P(\omega_2|x)$, $x\leftarrow\omega_1$. Prob. of error: $P(\omega_2|x)$.
-  If $P(\omega_1|x)<P(\omega_2|x)$, $x\leftarrow\omega_2$. Prob. of error: $P(\omega_1|x)$.
## 2.3.2 Loss Function