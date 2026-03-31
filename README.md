# 📉 Gradient Descent From Scratch

> *Building intuition for one of the most fundamental algorithms in machine learning — no black boxes, no shortcuts.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)

---

## 🧭 Overview

This project is a hands-on, visual exploration of **Gradient Descent** — the optimization algorithm at the heart of nearly every machine learning model. Built entirely from scratch using only **NumPy** and **Matplotlib**, it strips away the abstraction layers of modern ML libraries to show you exactly what's happening under the hood.

Across three progressively complex Jupyter notebooks, you'll watch gradient descent converge in real time — from minimizing a simple parabola to fitting a regression line to noisy data.

### Why learn gradient descent from scratch?

Most ML beginners jump straight to `sklearn` or `PyTorch` without understanding *why* models learn. Gradient descent is the engine that drives that learning. Understanding it deeply means you can:

- Debug models that aren't converging
- Tune learning rates with confidence
- Build intuition for loss landscapes and optimization
- Extend your knowledge to advanced optimizers (Adam, RMSProp, etc.)

---

## ✨ Features

- 📐 **Pure NumPy implementation** — no ML libraries, just math and arrays
- 📊 **Rich visualizations** — step-by-step plots showing descent in action
- 🧩 **Three levels of complexity** — 1D → 2D → real regression
- 🔰 **Beginner-friendly code** — heavily commented and easy to follow
- 📈 **Contour plot optimization paths** — see exactly how the algorithm navigates a loss surface
- 🎯 **Conceptual clarity** — each notebook pairs theory with working code

---

## 🗂️ Repository Structure

```
gradient-descent-from-scratch/
│── 1D_gradient_descent.ipynb
│── 2D_gradient_descent.ipynb
│── linear_regression_gradient_descent.ipynb
│── README.md
```

---

## 📓 Notebook Breakdown

### 1. `1D_gradient_descent.ipynb` — The Basics

**Purpose:** Minimize the function `f(x) = x²` using gradient descent.

**Key Concepts:**
- Derivatives as the slope of a function
- How the learning rate controls step size
- Convergence behaviour over iterations

**What You'll See:**
- A smooth parabola plotted in 2D
- A moving point descending step-by-step toward the minimum at `x = 0`
- A loss curve showing how `f(x)` decreases over iterations

---

### 2. `2D_gradient_descent.ipynb` — Into Higher Dimensions

**Purpose:** Minimize `f(x, y) = x² + y²` — a bowl-shaped surface in 3D space.

**Key Concepts:**
- Partial derivatives and gradients in 2D
- How gradient descent navigates a multi-dimensional surface
- Contour maps as a tool for visualizing optimization

**What You'll See:**
- A contour plot of the loss surface (concentric circles centered at the origin)
- An animated path of the optimization trajectory converging toward `(0, 0)`
- Intuition for how the algorithm "rolls downhill" in 2D

---

### 3. `linear_regression_gradient_descent.ipynb` — A Real ML Problem

**Purpose:** Fit a line `y = mx + b` to synthetically generated noisy data by learning the slope `m` and intercept `b` from scratch.

**Key Concepts:**
- Mean Squared Error (MSE) as a loss function
- Computing gradients with respect to model parameters
- Iterative parameter updates to minimize prediction error

**What You'll See:**
- A scatter plot of noisy data with the regression line evolving in real time
- A loss curve showing MSE decreasing over training epochs
- Final learned parameters compared to the true values

---

## 📐 How Gradient Descent Works

Gradient descent is an iterative optimization algorithm that minimizes a function by repeatedly moving in the direction of steepest descent — the **negative gradient**.

### The Intuition

Imagine you're standing on a hilly landscape blindfolded. Your goal is to reach the lowest valley. At each step, you feel the slope beneath your feet and take a step in the downhill direction. Gradient descent does exactly this — but in a mathematical parameter space.

### The Update Rule

At each iteration, parameters are updated using:

$$\theta = \theta - \eta \cdot \nabla J(\theta)$$

Where:

| Symbol | Meaning |
|---|---|
| `θ` | Model parameters (e.g., `x`, or `m` and `b`) |
| `η` (eta) | Learning rate — controls the step size |
| `∇J(θ)` | Gradient of the loss function with respect to `θ` |

```python
# Core update rule in code
theta = theta - learning_rate * gradient
```

### The Role of Learning Rate

| Learning Rate | Effect |
|---|---|
| Too large | Overshoots the minimum, may diverge |
| Too small | Converges very slowly |
| Just right ✅ | Smooth, stable convergence |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gradient-descent-from-scratch.git
cd gradient-descent-from-scratch
```

### 2. Install Dependencies

All you need is Python and three lightweight libraries:

```bash
pip install numpy matplotlib jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open your browser and navigate to the notebook you want to explore. We recommend starting with `1D_gradient_descent.ipynb` and working your way through in order.

---

## 📊 Visualizations

Each notebook produces clear, informative plots designed to build intuition:

### 🔵 Parabolic Descent (1D)

A 2D plot of `f(x) = x²` with the current position of the algorithm marked at each step. Watch the point slide down the curve and settle at the minimum.

### 🔴 Contour Path (2D)

A top-down contour map of `f(x, y) = x² + y²`, showing the optimization trajectory as a series of arrows spiraling inward toward the global minimum at the origin.

### 🟢 Regression Line Fitting

A scatter plot of data points with the learned regression line overlaid. As training progresses, you'll see the line rotate and shift until it fits the data well — visualizing learning in real time.

---

## 🎓 Learning Outcomes

By working through these notebooks, you will:

- ✅ Understand the mathematical foundations of gradient descent
- ✅ Implement the algorithm from scratch without any ML libraries
- ✅ Develop visual intuition for how optimization works in 1D and 2D
- ✅ Grasp the effect of learning rate on convergence
- ✅ Connect gradient descent to a real problem: linear regression
- ✅ Build the mental model needed to understand modern deep learning optimizers

---

## 🔮 Future Improvements

This project is designed to grow. Planned extensions include:

- 🎞️ **Animations** — Frame-by-frame animated descent using `matplotlib.animation`
- ⚡ **Stochastic & Mini-Batch Gradient Descent** — Introduce noise and compare convergence behaviour
- 🧠 **Neural Network Extension** — Apply gradient descent to a simple feedforward network with backpropagation
- 📉 **Momentum & Adam Optimizer** — Implement adaptive learning rate methods from scratch
- 🌄 **Non-Convex Loss Surfaces** — Explore local minima, saddle points, and escaping flat regions

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **NumPy** | Numerical computation and gradient calculations |
| **Matplotlib** | Plotting and visualization |
| **Jupyter Notebook** | Interactive, cell-by-cell execution environment |

---

<div align="center">

*Built with curiosity and a love for understanding things from first principles.*

⭐ **If this helped you understand gradient descent, consider starring the repo!** ⭐

</div>
