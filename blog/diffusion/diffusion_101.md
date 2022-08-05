# Diffusion 101


- Naive approach
	- sample $x_0$, roll out $x_t$ using transition kernel. 
	- train reverse model $p_\theta$ on pairs $x_{t-1},x_t$.
	- problem, not very efficient, within a rollout samples are not idependent (that the right way to put it, how does Markov property come into play here).
- Improved:
	- sample $x_{t-1}$ from $q_{t-1} = q(x_{t-1} \mid x_0 )$ and just roll out one step to get $x_t$.
	- still high variance, won't see that many samples that go through $x_t$
	- even better just samplt $x_t$ from $q_t$ and the compute the $\tilde q_t = q( x_{t-1} \mid x_t, x_0)$.




**Questions.**
- What is the right choice for $\sigma_t$ in the denoising model?
	- Why is $\tilde\beta$ not the right choice??
	- If the denoiser only approximates the right mean, would the right sigma differ from beta tilde?
- What is the relation/difference to normalizing flows etc...?