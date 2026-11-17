# ==========================================
# 1. Dataset Simulation
# ==========================================

from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt


# Create Two Datasets Simulations #

# Define total number of samples (n=10,000) and a fixed random seed for reproducibility
n = 10000
random_seed = 40 

# Generate the Linear dataset
# Two 2D Gaussian clusters centered far apart at (-5, -5) and (5, 5) with std=1.0 so they are linearly separable and do not overlap
X_lin, y_lin = make_blobs(n_samples=n, centers=[(-5, -5), (5, 5)], cluster_std=1.0, random_state=random_seed)

# Generate the Circular dataset
# Two concentric rings where the inner circle has half the radius of the outer circle (factor=0.5) and Gaussian noise (0.12) creates slight overlap between the two classes
X_circ, y_circ = make_circles(n_samples=n, factor=0.5, noise=0.12, random_state=random_seed)


# Visualization of the Simulated Datasets #

# Create a figure with 1 row and 2 side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Scatter plot of the Linear dataset colored by binary class label (y_lin)
axes[0].scatter(X_lin[:, 0], X_lin[:, 1], c=y_lin, s=5, alpha=0.6)
axes[0].set_title("Linear Dataset")
axes[0].set_xlabel("X1")
axes[0].set_ylabel("X2")

# Right subplot: Scatter plot of the Circular dataset colored by binary class label (y_circ)
axes[1].scatter(X_circ[:, 0], X_circ[:, 1], c=y_circ, s=5, alpha=0.6) 
axes[1].set_title("Circular Dataset")
axes[1].set_xlabel("X1")
axes[1].set_ylabel("X2")

# Adjust spacing so labels do not overlap and display the plots
plt.tight_layout()
plt.show()