### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 3f015e30-6141-11ec-2426-8fe9b0fd9896
md"

# Proximodistal Freezing and Freeing Degrees of Freedom (PDFF)

Many things to note here. Primarily this paper (cite it) involves the following:
 - PI^BB
 - others...

## $$\text{PI}^\text{BB}$$: Black-box Stochastic optim

The algorithm is as follows:
> **inputs:**
> - initial parameter vector $$\theta^\text{init}$$ 
> - cost function $$J: \theta \mapsto \mathbb{R}$$
> - exploration levels: $$\lambda^\text{init}$$, $$\lambda^\text{min}$$, $$\lambda^\text{max}$$
> - roll-outs per update: $$K$$
> - eliteness parameter: $$h$$
> **procedure:**
> - let $$\theta = \theta^\text{init}$$; $$\Sigma = \lambda^\text{init}\mathbf{I}$$
> - **while** cost not converged **do**
>   - *Exploration: sample parameters*
>   - **foreach** $$k$$ in $$K$$ **do**
>       - let $$\theta_k \sim \mathcal{N}(\theta, \Sigma)$$; $$J_k = J(\theta_k)$$ $$\triangleright$$ *cost of iterations*
>   - **end**
>   - let $$J^\text{min} = \min{\{J_k\}_{k=1}^K}$$; $$J^\text{max} = \max{\{J_k\}_{k=1}^K}$$
>   - *Evaluation: compute weight for each sample*
>   - **foreach** $$k$$ in $$K$$ **do**
>       - let $$P_k = \frac{\exp\left(-h\frac{J_k - J^\text{min}}{J^\text{max} - J^\text{min}}\right)}{\sum_{l=1}^K\exp\left(-h\frac{J_l - J^\text{min}}{J^\text{max} - J^\text{min}}\right)}$$ $$\triangleright$$ *relative probabilities for iterations*
>   - **end**
>   - *Update: weighted averaging over $$K$$ samples*
>   - define $$\mathbf{\Sigma} \gets \sum_{k=1}^{K}[P_k(\theta_k-\theta)(\theta_k-\theta)^\intercal]$$
>   - set $$\mathbf{\Sigma} \gets \texttt{boundcovar}(\mathbf{\Sigma}, \lambda^\text{min}, \lambda^\text{max})$$
>   - define $$\theta_\text{new} \gets \sum_{k=1}^K [P_k \theta_k]$$
> - **end**

"

# ╔═╡ Cell order:
# ╟─3f015e30-6141-11ec-2426-8fe9b0fd9896
