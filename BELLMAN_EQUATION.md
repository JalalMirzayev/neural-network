# Bellman Equation

$$V_\pi(s) = \sum_a\pi(a|s)\sum_{s'}\sum_{r}\left[r + \gamma V_\pi(s')\right]P(s',r|a,s)$$

# Derivation

We will first use the recursive nature of the total return $G_t$ and then apply laws for the expectation (linearity & constant)

$$
\begin{align}
V_\pi(s)&=\mathbb{E}\left[G_t |S_t = s\right]\\
&=\mathbb{E}\left[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots | S_t = s \right]\\
&=\mathbb{E}\left[R_{t+1} + \gamma \left(R_{t+2} + \gamma R_{t+3} + \ldots\right) | S_t = s \right]\\
&=\mathbb{E}\left[R_{t+1} + \gamma G_{t+1} | S_t = s \right]\\
&=\mathbb{E}\left[R_{t+1}| S_t = s \right] + \gamma \mathbb{E}\left[G_{t+1} | S_t = s \right]\\
\end{align}
$$

This is a partitioning of the State-Value function into a piece for the immediate reward and the discounted future reward.

## Immediate reward

The expected value is defined as
$$\mathbb{E}\left[A\right] = \sum_a a p(a) = \int a p(a) da$$

$$
\begin{align}
\mathbb{E}\left[R_{t+1}| S_t = s \right] &= \sum_r rP(r|S_t = s)
\end{align}
$$

The probability $P(r|S_t=s)$ is hiding some actual details. As we are looking at $R_{t+1}=r$ this is the reward of the next state $s'$. We want to de-marginalize the expression. By using the following law.

$$\sum_x p(x,y) = p(y)$$

$$
\begin{align}
\mathbb{E}\left[R_{t+1}| S_t = s \right] &= \sum_r rP(r|S_t = s)\\
&= \sum_r r\sum_a \sum_{s'}P(r,a,s'|S_t = s)
\end{align}
$$

Now apply conditional probability

$$
\begin{align}
\mathbb{E}\left[R_{t+1}| S_t = s \right] &= \sum_r r\sum_a \sum_{s'}P(r,a,s'|S_t = s)\\
&=\sum_r r\sum_a \sum_{s'}P(s', r|S_t = s, A_t = a)P(a|S_t = s)\\
&=\sum_r r\sum_a \sum_{s'}P(s', r|S_t = s, A_t = a)\pi(a|s)\\
&=\sum_r \sum_a \sum_{s'}r P(s', r|s, a) \pi(a|s)\\
\end{align}
$$

## Future reward

The future reward requires to calculate the return $G_{t+1} = R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \ldots$.
Hence, we need to de-marginalize on $a, s'$, and $r$. As $a, s', r$ have to have happened such that we.

$$
\begin{align}
\mathbb{E}\left[G_{t+1} | S_t = s \right] &= \sum_g g P(g|S_t=s)\\
&= \sum_g g \sum_{a, s', r}P(g, a, s', r |S_t=s)\\
&= \sum_g g \sum_{a, s', r}P(g|s, a, s', r)P(a, s', r|s)\\
\end{align}
$$

With the Markov Property
$$P(x_t| x_{t-1}, x_{t-2}, x_{t-3}, \ldots) = P(x_t|x_{t-1})$$
we can eliminate $s$, $a$, and $r$ from the first probability. This is possible as $G_{t+1}=g$ is referencing state $s'$ and its transition to $s''$. Hence, we do not care about $s$ the action $a$ which took us to $s'$ and the immediate reward $r$ which was released immediately before $s'$.

$$
\begin{align}
\mathbb{E}\left[G_{t+1} | S_t = s \right] &= \sum_g g \sum_{a, s', r}P(g|s, a, s', r)P(r, s', a|s)\\
&= \sum_g g \sum_{a, s', r}P(g|s')P(r, s', a|s)\\
&= \sum_g g \sum_{a, s', r}P(g|s')P(s', r| a,s)\pi(a|s)\\
&= \sum_{a, s', r}\sum_g g P(g|s')P(s', r| a,s)\pi(a|s)\\
&= \sum_{a, s', r}V_\pi(s')P(s', r| a,s)\pi(a|s)\\
\end{align}
$$

## Combine immediate and future reward

$$
\begin{align}
V_\pi(s) &=\mathbb{E}\left[R_{t+1}| S_t = s \right] + \gamma \mathbb{E}\left[G_{t+1} | S_t = s \right]\\
&= \sum_r \sum_a \sum_{s'}r P(s', r|s, a) \pi(a|s) + \gamma\sum_{a, s', r}V_\pi(s')P(s', r| a,s)\pi(a|s)\\
&=\sum_{a, s', r}\left(r  + \gamma V_\pi(s')\right) P(s', r|s, a) \pi(a|s)\\
&=\sum_a \pi(a|s)\sum_{s', r}\left(r  + \gamma V_\pi(s')\right) P(s', r|s, a) \\
\end{align}
$$

# Optimality for State-Value function $V_\pi(s)$

$$
\begin{align}
V_\pi(s) &=\sum_a \pi(a|s)\sum_{s', r}\left(r  + \gamma V_\pi(s')\right) P(s', r|s, a) \\
&=\sum_a \pi(a|s)Q(s, a) \\
\end{align}
$$

The goal is to choose $\pi$ such that we maximize the State-Value function $V_\pi(s)$.

$$V^*(s) = \max_{\pi}V_\pi(s)$$

But this is unfeasible as we would need to try out every possible action in every possible state. An alternative approach is visible if we notice that $\pi(a|s) \in [0, 1]$. If we have an optimal policy we will obviously only choose one action given a state which implies that $\pi(a|s) = 1$, while all other actions have a probability of 0.
With this we are left only with maximizing $Q(s,a)$.

$$
\begin{align}
V_\pi(s) &=\sum_a \pi(a|s)\sum_{s', r}\left(r  + \gamma V_\pi(s')\right) P(s', r|s, a) \\
&=\sum_a \pi(a|s)Q(s, a) \\
V^*(s)&=\max_a Q(s,a)\\
&=\max_a \sum_{s', r}\left(r  + \gamma V^*(s')\right) P(s', r|s, a)
\end{align}
$$

# Optimality for Action-Value function $Q_\pi(a, s)$

$$
\begin{align}
Q_\pi(a,s) &=\sum_{s', r}\left(r  + \gamma V_\pi(s')\right) P(s', r|s, a) \\
&=\sum_{s', r}\left(r  + \gamma \sum_{a'}\pi(a'|s')Q_\pi(a',s')\right) P(s', r|s, a) \\
Q^*(a,s) &= \max_\pi Q_\pi(a,s)\\
&= \sum_{s', r}\left(r  + \gamma \max_{a'}Q^*(a',s')\right) P(s', r|s, a) \\
\end{align}
$$

# Policy Iteration

$$
\begin{align}
V_\pi(s) &=\sum_a \pi(a|s)\sum_{s', r}\left(r  + \gamma V_\pi(s')\right) P(s', r|s, a) \\
&=\sum_a \pi(a|s)Q_\pi(a, s) \\
V^*(s) &=\max_a \sum_{s', r}\left[r  + \gamma V^*(s')\right] P(s', r|s, a)\\
Q^*(a,s) &= \sum_{s', r}\left[r  + \gamma \max_{a'}Q^*(a',s')\right] P(s', r|s, a) \\
\end{align}
$$

## Iteration methods

### Policy Iteration based on Bellman Expectation Equation

#### Policy Evaluation

$$
\begin{align}
V_{\pi, 1}(s) &= \sum_a\pi(a|s)\sum_{s'}\sum_r\left[r + \gamma V_{\pi, 0}(s') \right]P(s',r|a,s)\\
V_{\pi, 2}(s) &= \sum_a\pi(a|s)\sum_{s'}\sum_r\left[r + \gamma V_{\pi, 1}(s') \right]P(s',r|a,s)\\
V_{\pi, 3}(s) &= \sum_a\pi(a|s)\sum_{s'}\sum_r\left[r + \gamma V_{\pi, 2}(s') \right]P(s',r|a,s)\\
\end{align}
$$

We repeat this until we have

$$
\max_s\left|V_{\pi,k+1}(s) - V_{\pi, k}(s)\right| < \epsilon
.
$$

#### Policy Improvement

$$
\begin{align}
\pi'(s) &= \text{arg}\max_a\sum_{s'}\sum_r\left[r + \gamma V_\pi(s') \right]P(s', r|a, s)\\
&= \text{arg}\max_a Q_\pi(a, s)
\end{align}
$$

- Initialize Random Policy
- Initialize Values as 0
  - Policy Evaluation
    - Compute Values of the policy
  - Policy Improvement
    - Update your Policy
  - Repeat until convergence

### Bellman Optimality Equation

Value Iteration

- Initialize Values as 0
  - Compute optimal values
- Derive Policy from optimal values

# References

[Priyam Mazumdar (2025), Mathing the Bellman Equation: Derivation of the Foundational Equation in Reinforcement Learning](https://www.youtube.com/watch?v=4YXM7vEuR5c)
