**Diagnosing Heterogeneity in Prion Aggregates: Why Single-Exponential Fits Fail**\
Daniel Sprague


Problem Statement: We observe that an exponential decay rate model appears to underestimate the remaining population as the time $t$ increases.

Hypothesis: The population is a non-homogenous mixture of subpopulations with differing decay rates.

Result: We prove that a least squares estimate for a homogenous decay rate in a heterogeneous population will always overestimate $N(t)$ at early time points and significantly underestimate $N(t)$ for late timepoints.

Conclusion: This proof can be used to infer whether a sample is heterogenous or homogenous. 

<hr/>

**Prelude: Averaging Decay Rates from a Mixture Always Underestimates $N(t)$**

This section motivates the analysis of regressed decay rate estimates from hetergenous populations. Here, we assume we have a known mixture of hetergenous protein aggregates with known decay rates.

Given the mixture components, which sum to 1:

$$\boldsymbol{\pi} = [\pi_1, \dots, \pi_K] \in \mathbb{R}^K_{>0}, \quad \text{where} \quad \sum_{k=1}^K \pi_k = 1$$

And a vector of decay rates

$$\boldsymbol{\lambda} = [\lambda_1,\dots,\lambda_K] \in \mathbb{R}^K_{>0}$$

We can calculate $N(t)$ two different ways.

1. $$ \bar{\lambda} = \sum_k{\pi_k\lambda_k} = \boldsymbol{\pi}^\top\boldsymbol{\lambda}$$
$$N_{avg}(t) = e^{-\bar{\lambda}t}$$


2. $$N_{mix}(t) = \sum_{k}\pi_ke^{-\lambda_kt}$$


Jenson's inequality states that, for convex functions and a random variable $X$, the expectation of a function is greater than or equal to a function of the expectation.

$$\mathbb{E}[\psi(X)] \geq \psi(\mathbb{E}[X])$$

1. $\lambda$ is the random variable.

2. $\mathbb{E}[\lambda] = \bar{\lambda}$. 

3. $\psi(\lambda)$ is the exponential function $e^{-\lambda t}$ which is stated without proof to be a convex function. 

4. $\psi({\mathbb{E}[X]}) = e^{-\bar{\lambda}t}$

5. $\mathbb{E}[\psi(\lambda)] = \sum_k{\pi_k e^{-\lambda_kt}}$

Therefore, 

$$ \sum_k{\pi_k e^{-\lambda_kt}} \geq  e^{-\bar{\lambda}t}$$

$$N_{mix}(t) \geq N_{avg}(t) \quad \text{for all} \quad t.$$


<hr/>

**Behavior of Homogenous Least Squares Half Life Estimates in Heterogenous Populations**

Two key facts are necessary to set the scene for this analysis:

1. Exponential decay is strictly log-linear in semi-log space. This is stated without proof.
2. A convex mixture of exponential decays function is strictly log-convex in semi-log space.

**Proof of Log-Convexity**

$$L(t) = \ln N(t)$$

$$N(t) = \sum_k{\pi_k e^{-\lambda_kt}}$$

$$N'(t) = -\sum_k \lambda_k{\pi_k}e^{-\pi_k t}$$

Applying the chain rule and derivative of $\ln x$

$$L'(t) = \frac{N'(t)}{N(t)} = \frac{-\sum_i{\lambda_i(\pi_i}e^{-\lambda_i t})}{\sum_k{\pi_k e^{-\lambda_kt}}}$$

The term $\pi_i e^{-\lambda_i t}$ in the numerator corresponds exactly to the remaining proportion of mixture component $i$ in the overall population at time $t$. The denominator is simply $N(t)$. This then forms a probability distribution of proportions at each time point:

$$w_i(t) = \frac{\text{Amount of } i \text{ remaining}}{\text{Total Amount remaining}} = \frac{\pi_i e^{-k_i t}}{\sum_{j=1}^K \pi_j e^{-k_j t}} = \frac{\pi_i e^{-k_i t}}{N(t)}$$

Therefore, the derivative of $\ln N(t)$ is precisely equal to the weighted average of decay rates, where the weights correspond to proportion of each mixture component left at time $t$, pretty cool!!!

$$L'(t) = -\sum_i{w_i(t)\lambda_i}$$

To prove log-convexity, we need the second derivative.

$$ w'_i(t) = \frac{d}{dt}\left(\frac{\pi_i e^{-\lambda_i t}}{N(t)}\right)$$

$$w'_i(t) = \frac{N(t) \cdot \left(-\lambda_i\pi_ie^{-\lambda_i t}\right) - \left(\pi_ie^{-\lambda_i t}\right) \cdot N'(t)}{N(t)^2}$$

$$ w_i'(t) = -\lambda_i \underbrace{\frac{\pi_i e^{-\lambda_i t}}{N(t)}}_{w_i(t)} - \underbrace{\frac{\pi_i e^{-\lambda_i t}}{N(t)}}_{w_i(t)} \frac{N'(t)}{N(t)} $$

Recall $\frac{N'(t)}{N(t)} = L'(t)$, which is the weighted average rate at time $t$. We'll denote this weighted average rate as $\bar{\lambda}(t) = -L'(t)$.

$$w_i'(t) = -\lambda_i w_i(t) + w_i(t) \bar{\lambda}(t)$$

$$w_i'(t) = -w_i(t) (\lambda_i - \bar{\lambda}(t))$$


$$L''(t) = \frac{d}{dt} \left( -\sum_i w_i(t)\lambda_i \right) = -\sum_i \lambda_i w_i'(t)$$

Substitute our result for $w_i'(t)$:

$$L''(t) = -\sum_i \lambda_i \left[ -w_i(t) (\lambda_i - \bar{\lambda}(t)) \right]$$

$$L''(t) = \sum_i w_i(t) \lambda_i (\lambda_i - \bar{\lambda}(t))$$

Distribute the $\lambda_i$:

$$L''(t) = \sum_i w_i(t) \lambda_i^2 - \sum_i w_i(t) \lambda_i \bar{\lambda}(t)$$

Since $\bar{\lambda}(t)$ is a constant with respect to the summation index $i$, we can pull it out:

$$L''(t) = \underbrace{\sum_i w_i(t) \lambda_i^2}_{\mathbb{E}[\lambda^2]} - \bar{\lambda}(t) \underbrace{\sum_i w_i(t) \lambda_i}_{\mathbb{E}[\lambda]}$$


Since $\bar{\lambda}(t) = \mathbb{E}[\lambda]$ (the weighted mean):

$$L''(t) = \mathbb{E}[\lambda^2] - (\mathbb{E}[\lambda])^2$$

This is the definition of Variance:

$$L''(t) = \text{Var}_w(\lambda)$$


1. For $\text{Var}(\lambda) > 0$, we have proven the model is log-convex. 
2. If $\text{Var}(\lambda) \approx 0$, the model is not a mixture. This means we can perform a statistical test to assess whether the population is homogenous or heterogenous.
3. This relationship to variance also means that if the data is highly curved in semi-log space, the population is a mix of very fast and very slow subpopulations. 

We can also show that attempting to fit a single rate model to a mixture will result in a characteristic see-saw of residuals:


$f(t) = \ln \left( \sum \pi_i e^{-\lambda_i t} \right)$

$g(t) = \ln \left( e^{-\lambda t} \right) = -\lambda t$


$$D(t) = f(t) - g(t) = f(t) + \lambda t$$

$$D''(t) = f''(t) - g''(t)$$

We know $g(t) = -\lambda t$ is a line, so $g''(t) = 0$.

We proved earlier that $f''(t) = \text{Var}_w(\lambda) > 0$ (assuming heterogeneity exists).


$$D''(t) > 0 \quad \text{for all } t$$

$D(t)$ is strictly convex

A strictly convex function can intersect the horizontal axis $D(t)=0$ at most twice.

Next, we can use the properties of least squares to proove that the single rate least squares must cross the mixture of rates.

$$S(\lambda) = \sum_{j=1}^{M} \left( \underbrace{e^{-\lambda t_j}}_{\text{Model}} - \underbrace{\sum_{k=1}^K \pi_k e^{-\lambda_k t_j}}_{\text{Truth}} \right)^2$$

Least squares solves for where the derivative with respect to the parameter $\lambda$ is zero.

$$\frac{dS}{d\lambda} = \sum_{j=1}^{M} 2 \left( e^{-\lambda t_j} - \sum_{k=1}^K \pi_k e^{-\lambda_k t_j} \right) \cdot \frac{d}{d\lambda} \left( e^{-\lambda t_j} - \sum_{k=1}^K \pi_k e^{-\lambda_k t_j} \right)$$

$$0 = \sum_{j=1}^{M} D(t) \cdot \underbrace{(t_j e^{-\lambda t_j})}_{\text{Weight } w_j}$$

The weights term in the summation is strictly positive. Therefore, for the sum to be zero the difference function $D(t)$ must switch signs at least once, and we already know that $D(t)$ is a parabola. Therefore, the single rate model is a secant approximation to the mixture.

This enables us to apply the Wald-Wolfowitz Runs Test on the signs of the difference function $D(t)$ to test the hypothesis that the data is heterogenous.