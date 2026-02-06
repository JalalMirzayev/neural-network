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

$$
\begin{align}
\mathbb{E}\left[G_{t+1} | S_t = s \right] &= \sum_g g P(g|S_t=s)\\
&= \sum_g g \sum_{a, s', r}P(g, |S_t=s)\\
&= \sum_g g \sum_{a, s', r}P(g|s, a, s', r)P(r, s', a|s)\\
\end{align}
$$

With the Markov Property
$$P(x_t| x_{t-1}, x_{t-2}, x_{t-3}, \ldots) = P(x_t|x_{t-1})$$
we can eliminate $s$, $a$, and $r$ from the first probability.

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

# References

[Priyam Mazumdar (2025), Mathing the Bellman Equation: Derivation of the Foundational Equation in Reinforcement Learning](https://www.youtube.com/watch?v=4YXM7vEuR5c)
