---
layout: distill
title: Introduction to Stochastic Interpolants
description: 
  Prominent generative modeling frameworks such as Flow Matching and score-based Diffusion Models establish a smooth transformation between a Gaussian distribution and a data distribution. In this blog post, we provide an introduction to the more general framework of Stochastic Interpolants, which allows one to flexibly interpolate between any two distributions and learn a velocity field to transform samples from one into samples of the other. No prior knowledge of generative models is required for this introduction.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

bibliography: 2026-04-27-introduction-to-stochastic-interpolants.bib

toc:
  - name: Background
    subsections:
      - name: Notation
  - name: Methods
    subsections:
      - name: Building a stochastic interpolant model
      - name: Deriving an expression for the velocity field 
      - name: Deriving the loss function
  - name: Connection to other generative models
  - name: Conclusion
---
## Background

In generative modeling frameworks such as flow matching <d-cite key="lipman2022flow"></d-cite> and score-based diffusion models <d-cite key="song2020score"></d-cite>, one constructs a smooth transition from a Gaussian to a data distribution. This transition is governed by a velocity field that operates on the samples by pushing them from the Gaussian to the target. To generate new data samples, we begin by drawing from the latent noise distribution and then use a learned velocity field to iteratively move towards regions where the data distribution has high probability mass. The result is a synthetic sample from the approximate data distribution. In such generative models, the velocity field is a parameterized model learned from data.

The goal of the stochastic interpolant framework <d-cite key="albergo2023stochastic, albergo2022building"></d-cite> is more general. Here we aim to interpolate between any two densities. The stochastic interpolant framework provides a flexible family of possible interpolations. After selecting the form of the interpolation, we can train a corresponding velocity field that gradually transforms samples from one distribution into the other.

In the context of generative modeling, this corresponds to transforming noise into data: one distribution represents the latent noise, and the other corresponds to the data distribution. However, the framework goes beyond just flow matching and score-based diffusion, making it applicable to tasks other than generating data from noise.

### Notation

Throughout the blog post, we use the following conventions:

- Boldface letters (e.g., $$\mathbf{x}$$) denote column vectors in $$\mathbb{R}^d$$, whereas non-bold letters (e.g., $t$) denote scalar quantities in $$\mathbb{R}$$
- We write $$\partial_t := \tfrac{\partial}{\partial t}$$ for the partial derivative with respect to time.
- We denote by $$\nabla_{\mathbf{x}} := (\partial_{\mathbf{x}_1}, \ldots, \partial_{\mathbf{x}_d})$$ the gradient operator with respect to the spatial variables.
- Throughout the blog post we abbreviate $$\mathcal{N}(\mathbf{x};\boldsymbol{\mu}, \sigma^2)$$ for the density at point $$\mathbf{x}$$ of the $$d$$-dimensional normal distribution $$\mathcal{N}(\boldsymbol{\mu}, \sigma^2)$$ with mean $\boldsymbol{\mu}$ and a covariance matrix with main diagonal entries of $$\sigma^2$$ and else $$0$$.
- We write $$\delta(\mathbf{x} - \boldsymbol{\mu})$$ for the delta distribution in $$\mathbb{R}^d$$ where all the mass is concentrated at $$\boldsymbol{\mu}$$. Intuitively, the delta distribution can be understood as the limit $$ \mathrm{lim}_{\sigma \rightarrow 0} \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2)$$.
  
## Methods

We begin with a short three-step overview of how to build a stochastic interpolant model. Afterwards, we will prove why this approach works.

### Building a stochastic interpolant

To construct a stochastic interpolant, we need access to samples drawn proportional to some densities $$p_0$$ and $$p_1$$ that we want to connect. Throughout this blog, we restrict ourselves to densities defined on the $$d$$-dimensional real space $$\mathbb{R}^d$$.

#### 1. Define intermediate densities

To define the intermediate densities $$p_t$$ (for all $$t \in [0, 1]$$) between $$p_0$$ and $$p_1$$, we introduce two functions: an *interpolant* $$\,\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1)$$ and a *noise amplitude function* $$\gamma(t)$$. A natural choice for the *interpolant* is to use a simple convex combination $$\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1,$$ of the two samples that creates a line segment between them. A reasonable choice for the *noise amplitude function* is to have the noise contribution peak midway through the interpolation and vanish at the boundaries, e.g. $$\gamma(t) = t(1 - t).$$

This means our intermediates can now be written algebraically as

$$
p_t(\mathbf{x}) = \int p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) p_0(\mathbf{x}_0) p_1(\mathbf{x}_1) p_\mathcal{N}(\mathbf{z}) \mathrm{d}\mathbf{x}_0 \mathrm{d}\mathbf{x}_1 \mathrm{d}\mathbf{z}
$$

with
$$p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) = \delta(\mathbf{x} - (\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}))$$
and
$$p_\mathcal{N}(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, 1).$$ Although this formulation of $$p_t$$ might look intimidating at first, drawing samples from it is straightforward. We can sample $$\mathbf{x}_t \sim p_t$$ by drawing $$\mathbf{x}_0 \sim p_0$$, $$\mathbf{x}_1 \sim p_1$$, and $$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, 1)$$, then setting
$$\mathbf{x}_t = \mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}.$$

When choosing the *interpolant* and the *noise amplitude function*, we must ensure that the boundary conditions hold:

$$
p_0(\mathbf{x}) = \int p_0(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) p_0(\mathbf{x}_0) p_1(\mathbf{x}_1) p_\mathcal{N}(\mathbf{z}) \mathrm{d}\mathbf{x}_0 \mathrm{d}\mathbf{x}_1 \mathrm{d}\mathbf{z},
$$

$$
p_1(\mathbf{x}) = \int p_1(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) p_0(\mathbf{x}_0) p_1(\mathbf{x}_1) p_\mathcal{N}(\mathbf{z}) \mathrm{d}\mathbf{x}_0 \mathrm{d}\mathbf{x}_1 \mathrm{d}\mathbf{z}.
$$

A sufficient condition for this is that:
- the *interpolant* satisfies
$$\mathbf{I}(0, \mathbf{x}_0, \mathbf{x}_1) = \mathbf{x}_0,\, \mathbf{I}(1, \mathbf{x}_0, \mathbf{x}_1) = \mathbf{x}_1$$
- and that the *noise amplitude function* fulfills
$$\gamma(0) = 0 = \gamma(1)$$, such that the noise vanishes at the start and end of the interpolantion.

{% include figure.liquid path="assets/img/2026-04-27-introduction-to-stochastic-interpolants/pt.png" class="img-fluid"
caption="Interpolation between two 1D Gaussian Mixture Models. Each $p_t$, defined by choosing an interpolant and a noise function, has an associated velocity field $\mathbf{b}_t$ that we aim to learn." %}

#### 2. Training

We learn a parametric model $$\mathbf{b}_{\boldsymbol{\theta}}$$ of the velocity field for $$p_t$$, which we defined in the first step, by solving the following optimization problem:

$$
\min_{\boldsymbol{\theta}} \underset{\substack{t \sim \mathcal{U}[0, 1] \\ \mathbf{x}_0 \sim p_0 \\ \mathbf{x}_1 \sim p_1 \\ \mathbf{z} \sim \mathcal{N}(\mathbf{0}, 1) \\\mathbf{x}_t = \mathbf{I}(t,\mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}}}{\mathbb{E}} \big|\!\big| \underbrace{\partial_t \mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1)}_{\text{ e.g. } \mathbf{x}_1 -\mathbf{x}_0} + \underbrace{\partial_t\gamma(t)}_{\text{and } 1 - 2t \,}\mathbf{z} - \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\big|\!\big|^2.
$$

As in other areas of deep learning, this can be achieved using **stochastic gradient descent (SGD)**. We follow the gradient with respect to the parameters $$\boldsymbol{\theta}$$ based on a Monte Carlo estimate of the objective function. This is done by selecting a mini-batch size and replacing the full expectation with an empirical average over the batch. In practice, the training loss can be implemented as follows:
```python
def loss(b_model, x0, x1, t):
  """
    x0, x1: of shape (batch_size, d)
    t:      of shape (batch_size, 1)
  """
  z = randn_like(x0)
  xt = (1-t) * x0 + t * x1 + gamma(t) * z
  I_dt = x1 - x0
  gamma_dt = 1 - 2 * t
  target = I_dt + gamma_dt * z
  return ((target - b_model(xt, t))**2).mean()
```
The loss is fully differentiable (as long as the parametric model itself is differentiable). Therefore, we can use any automatic differentiation library—such as PyTorch, JAX, or TensorFlow—to compute gradients with respect to the model parameters.

#### 3. Inference

To transform a sample $$\mathbf{x}_0$$ from $$p_0$$ into a sample from $$p_1$$, we follow the estimated velocity field by simulating the **ordinary differential equation (ODE**):

$$
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)
$$

from $$t = 0$$ to $$t = 1$$.  
For the simulation, we can use **Euler’s method** and iterate

$$
\mathbf{x}_{t + \Delta t} \leftarrow \mathbf{x}_{t} + \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\, \Delta t, \qquad t \leftarrow t + \Delta t
$$

for $$S > 0$$ steps with step size $$\Delta t = 1/S$$.

We can also simulate the ODE in reverse time. Moreover, we are not restricted to starting or ending at the boundaries: we can choose any two time points $$t_\text{start}$$ and $$t_\text{end} \in [0, 1]$$ and transform samples from $$p_{t_\text{start}}$$ into samples from $$p_{t_\text{end}}$$.

In code we can implement this as follows:

```python
def euler(b_model, x_start, t_start, t_end, S):
  x = x_start
  timesteps = linspace(t_start, t_end, S)
  dt = (t_end - t_start) / S
  for t in timesteps:
    x = x + b_model(x, t) * dt
  x_end = x
  return x_end
```

Score-based diffusion models also allow for continuously transforming samples using a **stochastic differential equation (SDE)**. Although the stochastic interpolant framework includes this option, in this blog post we focus on performing inference via the ODE formulation.

#### Example

In the following animation, we can see two stochastic interpolant options between two 1D bimodal Gaussian mixture models, where each Gaussian has a standard deviation of $$0.2$$.

<div class="m-page">
  <iframe src="{{ 'assets/html/2026-04-27-introduction-to-stochastic-interpolants/pt_2combo.html' | relative_url }}" style="min-width:650px; min-height:750px;"
  frameborder="0"
  scrolling="no"></iframe>
</div>

Each of the two plots shows three key elements:  
1. The **density values** of $$p_t$$ across space–time, where yellow regions indicate areas of high density and black regions correspond to density values close to zero.  
2. **Cyan arrows** representing the velocity field $$(\Delta t, \Delta t \mathbf{b}_t(\mathbf{x}))$$.  
3. **White lines** showing trajectories $$\mathbf{x}(t)$$ through space–time generated by integrating along this velocity field.

Across the animation frames, the strength of the *noise amplitude function* $$\gamma(t)$$ changes, allowing us to see how this affects both the vector field and the resulting trajectories. In our 1D example we can see that a small noise amplitude can help to make the trajectories straighter. Trajectories with gentler curvature are easier to simulate during inference. At each step, Euler's method assumes the path is locally straight, so when the trajectory bends sharply, this approximation breaks down. To capture those sharper turns accurately, we need to increase the number of inference steps $$S$$. Therefore, we should design our interpolant to produce trajectories that are as straight as possible during inference.    

---

But why is all of this possible? Why can we generate a sample from $$p_{t_\text{end}}$$ at any $$t_\text{end} \in [0, 1]$$ as long as we start with a sample $$\mathbf{x}_{t_\text{start}}$$ from $$p_{t_\text{start}}$$ and follow the velocity field $$\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)$$ until $$t_\text{end}$$?

To understand why this works, we first need an algebraic formulation of the **true velocity field** $$\mathbf{b}_t(\mathbf{x})$$ that generates $$p_t$$. We will see that this leads to an expression for $$\mathbf{b}_t(\mathbf{x})$$ that is generally intractable in the interesting cases. To handle this intractability, we therefore train a **surrogate model** $$\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)$$ that approximates $$\mathbf{b}_t$$ in feasible time.

### Deriving an Expression for the Velocity Field

A velocity field $$\mathbf{b}_t(\mathbf{x})$$ that generates $$p_t$$ must obey the **continuity equation**:

$$
\partial_t p_t(\mathbf{x}) = -\nabla_{\mathbf{x}} \cdot \big(p_t(\mathbf{x})\,\mathbf{b}_t(\mathbf{x})\big).
$$

The continuity equation links a time-dependent density to its corresponding vector field. 
- On the left-hand side, it expresses how the density $$p_t(\mathbf{x})$$ changes over time — the *rate of change* of probability mass at point $$\mathbf{x}$$.  
- On the right-hand side, it measures how much of the flux represented by the weighted velocity field $$p_t(\mathbf{x})\,\mathbf{b}_t(\mathbf{x})$$ diverges/flows from that point.

{% include figure.liquid path="assets/img/2026-04-27-introduction-to-stochastic-interpolants/divergence.png" class="img-fluid"
caption="This visualization illustrates how the divergence operator $\nabla_{\mathbf{x}} \cdot $ quantifies local expansion or compression of a flux field. Each panel shows a different flux $p_t(\mathbf{x})\, \mathbf{b}_t(\mathbf{x})$ in two dimensions. <strong>Left:</strong> A positive divergence indicates a source region, where flux flows outward. For such a flux, the entropy of $p_t$ increases over time. <strong>Middle:</strong> Zero divergence represents a rotational or incompressible flux, where local inflow and outflow balance. Here, $p_t$ remains constant in time. <strong>Right:</strong> A negative divergence corresponds to a sink region, where flux converges inward. In this case, the entropy of $p_t$ decreases over time." %}

Intuitively, if a large amount of probability mass from neighboring regions flows into $$\mathbf{x}$$ with high velocity, the local density increases. Conversely, if more mass flows out of $$\mathbf{x}$$ than flows in, the density there decreases. In this way, the continuity equation ensures that total probability is conserved as the density moves in space and evolves over time.

To find a formula for $$\mathbf{b}_t$$, we start from the left side of the continuity equation and transform it into the right-hand side. Plugging in the definition of $$p_t$$ and swapping $$\int$$ with $$\partial_t$$ gives:

$$
\partial_t p_t(\mathbf{x}) = \int \partial_t p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}.
$$

Inside the integral, we find the partial rate of change $$\partial_t p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})$$ given $$\mathbf{x}_0$$, $$\mathbf{x}_1$$, and $$\mathbf{z}$$. This conditional distribution is a delta peak whose mass at time $$t$$ is concentrated at $$\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}$$, defining a path through time.  

{% include figure.liquid path="assets/img/2026-04-27-introduction-to-stochastic-interpolants/partial_just_one.png" class="img-fluid" %}

The velocity along this path is simply $$\partial_t \mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \partial_t \gamma(t)\mathbf{z}$$. By following this velocity over time, we move along the path. Thus, for the conditional distribution, the continuity equation holds:

$$
\partial_t p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) = \partial_t \delta(\mathbf{x} - (\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z})) \qquad \qquad \qquad \qquad \qquad \qquad  
$$

$$ 
\qquad \qquad \qquad \quad  = -\partial_t (\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}) \cdot\nabla_\mathbf{x} \delta(\mathbf{x} - (\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}))$$

$$ 
\qquad \qquad \qquad \qquad = - \nabla_\mathbf{x} \cdot \big(\delta(\mathbf{x} - (\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z})) \partial_t (\mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}) \big)
$$

$$
=- \nabla_\mathbf{x} \cdot \big(p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\,\mathbf{b}_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\big),
$$

with

$$
\mathbf{b}_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) = \partial_t \mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \partial_t \gamma(t)\mathbf{z}.
$$

We can now use this equality to obtain
<div class="l-body-outset">
  <p>$$\partial_t p_t(\mathbf{x}) = \int \overbrace{- \nabla_\mathbf{x} \cdot \big(p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\,\mathbf{b}_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\big)}^{\partial_t p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})} p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}.$$
  </p>
</div>

Pulling $$- \nabla_\mathbf{x} \cdot$$ outside the integral yields:
<div class="l-body-outset">
  <p>$$\partial_t p_t(\mathbf{x}) = - \nabla_\mathbf{x} \cdot \bigg(\int p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\,\mathbf{b}_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}\bigg).$$
  </p>
</div>

Finally, we multiply inside the brackets by $$1 = \textcolor{blue}{p_t(\mathbf{x})} / \textcolor{blue}{p_t(\mathbf{x})}$$:

<div class="l-body-outset">
  <p>$$ \partial_t p_t(\mathbf{x}) = - \nabla_\mathbf{x} \cdot \bigg(\textcolor{blue}{p_t(\mathbf{x})} \underbrace{\int \mathbf{b}_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) \frac{p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})}{\textcolor{blue}{p_t(\mathbf{x})}}\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}}_{\mathbf{b}_t(\mathbf{x})}\bigg).$$</p>
</div>

This gives us an algebraic formulation of the velocity field that generates $$p_t(\mathbf{x})$$:

$$
\mathbf{b}_t(\mathbf{x}) = \int \mathbf{b}_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) \overbrace{\frac{p_t(\mathbf{x}\,\vert\, \mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})}{p_t(\mathbf{x})}}^{p(\mathbf{x}_0, \mathbf{x}_1, \mathbf{z}\,|\, \mathbf{x})}\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}.
$$

Despite its intractable and complex form, this integral provides a way to derive a loss function for training the surrogate model $$\mathbf{b}_{\boldsymbol{\theta}}$$. This is the focus of the next section.

{% include figure.liquid path="assets/img/2026-04-27-introduction-to-stochastic-interpolants/partial.png" class="img-fluid"
caption="$p_t(\mathbf{x})$ is the mean of all partial densities $p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})$ with respect to $p(\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})$, and the velocity field $\mathbf{b}_t(\mathbf{x})$ is the mean of all partial velocity fields $\mathbf{b}_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})$ over $p_t(\mathbf{x}_0, \mathbf{x}_1, \mathbf{z} \,|\, \mathbf{x})$." %}

### Deriving the Loss Function

We now aim to design an objective function for which $$\mathbf{b}_t(\mathbf{x})$$ is the optimal solution. Whether we can actually reach this optimum depends, of course, on the flexibility of the chosen parametric model.

Let us start by writing a weighted least-squares regression loss that, by construction, achieves its minimum at $$\mathbf{b}_t(\mathbf{x})$$:

$$
\min_{\boldsymbol{\theta}} \int \|\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t) - \mathbf{b}_t(\mathbf{x})\|^2\, p_t(\mathbf{x})\, \mathrm{d}\mathbf{x}\, \mathrm{d}t.
$$

However, this formulation is not suitable for SGD. To perform SGD, we need all integrals, including the marginalization integrals in the definition of $\mathbf{b}_t$, to be outside the $$L^2$$-norm. We therefore begin by decomposing the squared norm:

<div class="l-body-outset">
  <p>$$ \min_{\boldsymbol{\theta}} \int \|\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2\, p_t(\mathbf{x})\, \mathrm{d}\mathbf{x}\, \mathrm{d}t
- 2 \int \mathbf{b}_t(\mathbf{x})^\top \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)\, p_t(\mathbf{x})\, \mathrm{d}\mathbf{x}\, \mathrm{d}t
+ \underbrace{\int \|\mathbf{b}_t(\mathbf{x})\|^2\, p_t(\mathbf{x})\, \mathrm{d}\mathbf{x}\, \mathrm{d}t}_{c_1}.$$
  </p>
</div>

The last term, $$c_1$$, is independent of $$\boldsymbol{\theta}$$ and therefore constant. Next, we plug in the definitions of $$p_t(\mathbf{x})$$ and $$\mathbf{b}_t(\mathbf{x})$$:

<div class="l-body-outset">
  <p>$$ \min_{\boldsymbol{\theta}} \int \|\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2
\overbrace{\int p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\, \mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}}^{p_t(\mathbf{x})} \,
\mathrm{d}\mathbf{x}\, \mathrm{d}t$$
$$
- 2 \int \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)^\top
\underbrace{\int \mathbf{b}_t(\mathbf{x}\,\vert\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})
\frac{p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})}{p_t(\mathbf{x})}\,
\mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}}_{\mathbf{b}_t(\mathbf{x})} \,
p_t(\mathbf{x})\, \mathrm{d}\mathbf{x}\, \mathrm{d}t + c_1.
$$</p>
</div>

We now move all integrals to the outside, cancel the factor $$p_t(\mathbf{x}) / p_t(\mathbf{x})$$ in the second term, and add a constant for quadratic completion:
<div class="l-body-outset">
  <p>$$ \min_{\boldsymbol{\theta}} \int \textcolor{blue}{\|\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2}\,
p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\,
\mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}\, \mathrm{d}\mathbf{x}\, \mathrm{d}t
$$

$$
\textcolor{blue}{- 2} \int \textcolor{blue}{\mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)^\top \mathbf{b}_t(\mathbf{x}\,\vert\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})}\,
p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\,
\mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}\, \mathrm{d}\mathbf{x}\, \mathrm{d}t
$$

$$
+ \underbrace{\int \textcolor{blue}{\|\mathbf{b}_t(\mathbf{x}\,\vert\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\|^2}\,
p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\,
\mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}\, \mathrm{d}\mathbf{x}\, \mathrm{d}t}_{c_2}
+ c_1 - c_2.$$
</p>
</div>

Combining the terms highlighted in blue, we obtain:

<div class="l-body-outset">
  <p>$$ \min_{\boldsymbol{\theta}} \int
\textcolor{blue}{\|\mathbf{b}_t(\mathbf{x}\,\vert\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z}) - \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}, t)\|^2}\,
p_t(\mathbf{x}\,|\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})\, p_0(\mathbf{x}_0)\, p_1(\mathbf{x}_1)\, p_\mathcal{N}(\mathbf{z})\,
\mathrm{d}\mathbf{x}_0\, \mathrm{d}\mathbf{x}_1\, \mathrm{d}\mathbf{z}\, \mathrm{d}\mathbf{x}\, \mathrm{d}t
+ c_1 - c_2.$$
</p>
</div>

Finally, we can express this as an expectation and substitute the definition of the partial velocity field $$\mathbf{b}_t(\mathbf{x}\,\vert\,\mathbf{x}_0, \mathbf{x}_1, \mathbf{z})$$:

$$
\min_{\boldsymbol{\theta}} \underset{\substack{
t \sim \mathcal{U}[0, 1] \\
\mathbf{x}_0 \sim p_0,\ \mathbf{x}_1 \sim p_1 \\
\mathbf{z} \sim \mathcal{N}(\mathbf{0}, 1) \\
\mathbf{x}_t = \mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1) + \gamma(t)\mathbf{z}
}}{\mathbb{E}}
\big\|\partial_t \mathbf{I}(t, \mathbf{x}_0, \mathbf{x}_1)
+ \partial_t \gamma(t)\mathbf{z}
- \mathbf{b}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\big\|^2
+ c_1 - c_2.
$$

Since we are only interested in the argument $$\boldsymbol{\theta}$$ that minimizes the objective, rather than the actual loss value, we can safely ignore the constants $$+ c_1 - c_2$$ during optimization.

## Connection to Other Generative Models

Other generative modeling frameworks based on either flows or score-based diffusion can be described within the stochastic interpolant framework. Most generative models generate data samples—such as images—starting from a Gaussian noise distribution. When either $$p_0$$ or $$p_1$$ is Gaussian, there are multiple ways to define the schedule using different choices of *interpolants* and *noise amplitude functions*.

| Methods |$$ \mathbf{I}(t, \mathbf{x}_0, \mathbf{x_1})$$  | $$ \gamma(t)$$  | boundary conditions  |
|---|:---:|:---:|---|
| Karras EDM <d-cite key="karras2022elucidating"></d-cite>| $$\mathbf{x}_0$$ |  $$t \cdot t_\text{max}$$ | $$p_0 = p_\text{data}, p_1 = \mathcal{N}(\mathbf{0}, t_\text{max}^2)$$ |
|  | $$ \mathbf{x}_0 + t \mathbf{x}_1$$ | $$ 0$$ | $$p_0 = p_\text{data}, p_1 = \mathcal{N}(\mathbf{0}, t_\text{max}^2)$$ |
| Song VE <d-cite key="song2020score"></d-cite>| $$ \mathbf{x}_0 $$ | $$ \sqrt{t \cdot t_\text{max}}$$ | $$p_0 = p_\text{data}, p_1 = \mathcal{N}(\mathbf{0}, t_\text{max})$$ |
|  | $$ \mathbf{x}_0 + \sqrt{t} \, \mathbf{x_1}$$  | $0$ | $$p_0 = p_\text{data}, p_1 = \mathcal{N}(\mathbf{0}, t_\text{max})$$ |
| Flow Matching <d-cite key="lipman2022flow"></d-cite>| $$ t \mathbf{x}_1 $$| $$ (1-t) $$ | $$ p_0 = \mathcal{N}(\mathbf{0}, 1),p_1 = p_\text{data}$$  |
|  | $$ (1-t)\mathbf{x}_0 + t \mathbf{x}_1$$ | $0$ | $$ p_0 = \mathcal{N}(\mathbf{0}, 1),p_1 = p_\text{data}$$|
| 1-Rectified Flow <d-cite key="liu2022flow"></d-cite>| $$(1-t)\mathbf{x}_0 + t\mathbf{x}_1$$ | $$ 0 $$ | any $$ p_0, p_1$$ |

In the EDM schedule proposed by Karras et al., $$t_\text{max}$$ is chosen such that 

$$\mathbf{x}_1 = \mathbf{x}_0 + t_\text{max}\mathbf{z} \sim \mathcal{N}(\mathbf{0}, t_\text{max}^2), \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, 1).$$ 

In other words, $$t_\text{max}$$ is set high enough that the signal from $$\mathbf{x}_0$$ becomes negligible compared to the strong noise contribution. The same principle applies to the variance exploding (VE) schedule introduced by Song et al.

The stochastic interpolant framework thus provides a unifying and highly flexible perspective for describing and designing flow-based and score-based diffusion models—and even for extending beyond them.

## Conclusion

In this post, we explored the stochastic interpolant framework. By viewing the transformation between two distributions as a stochastic interpolation, we can naturally recover well-known models like Flow Matching and score-based Diffusion Models as special cases. This perspective not only clarifies the shared principles behind these methods but also offers the possibility to new designs that interpolate between any pair of distributions. Future research may build on this framework to explore richer interpolation schemes, alternative noise functions, or applications beyond generative models—where learning smooth, probabilistic transformations between arbitrary densities are central.
