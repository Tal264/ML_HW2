# ==========================================
# 5. Noise analysis
# ==========================================

# Import dataset generators from scikit-learn and plotting tools from matplotlib
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import numpy as np


# ==========================================
# Define Data Sets (Linear & Circular)
# ==========================================

# Define total number of samples (n=10,000) and a fixed random seed for reproducibility
n = 10000
random_seed = 40 

# Generate the Linear dataset:
# Two 2D Gaussian clusters centered far apart at (-5, -5) and (5, 5) with std=1.0
# so they are linearly separable and do not overlap
X_lin, y_lin = make_blobs(n_samples=n, centers=[(-5, -5), (5, 5)], cluster_std=1.0, random_state=random_seed)

# Generate the Circular dataset:
# Two concentric rings where the inner circle has half the radius of the outer circle (factor=0.5)
# and Gaussian noise (0.12) creates slight overlap between the two classes
X_circ, y_circ = make_circles(n_samples=n, factor=0.5, noise=0.12, random_state=random_seed)


# ==========================================
# Split both datasets into training and testing sets
# ==========================================
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=random_seed)
X_circ_train, X_circ_test, y_circ_train, y_circ_test = train_test_split(X_circ, y_circ, test_size=0.2, random_state=random_seed)

# ==========================================
# Helper function - label-shuffling function & loop over noise percentages
# ==========================================

def shuffle_labels(y, pct, seed):
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_shuffle = int(pct * len(y))
    idx = rng.choice(len(y), size=n_shuffle, replace=False)
    y_noisy[idx] = rng.permutation(y_noisy[idx])
    return y_noisy

# Define noise levels and empty lists to store test accuracies
noise_levels = [0.0, 0.15, 0.25, 0.50]
lin_accuracies = []
circ_accuracies = []

for pct in noise_levels:

    y_lin_noisy = shuffle_labels(y_lin_train, pct, random_seed)
    y_circ_noisy = shuffle_labels(y_circ_train, pct, random_seed)

    # Linear SVM on Linear dataset
    svm_lin = SVC(kernel="linear", random_state=random_seed)
    svm_lin.fit(X_lin_train, y_lin_noisy)
    lin_acc = svm_lin.score(X_lin_test, y_lin_test)
    lin_accuracies.append(lin_acc)

    # RBF SVM on Circular dataset
    svm_rbf = SVC(kernel="rbf", random_state=random_seed)
    svm_rbf.fit(X_circ_train, y_circ_noisy)
    circ_acc = svm_rbf.score(X_circ_test, y_circ_test)
    circ_accuracies.append(circ_acc)

    print(f"Noise {int(pct*100)}% -> Linear Acc: {lin_acc:.4f} | Circular Acc: {circ_acc:.4f}")

# ==========================================
# Plot performance
# ==========================================

noise_pcts = [p * 100 for p in noise_levels]
plt.figure(figsize=(8, 5))
plt.plot(noise_pcts, lin_accuracies, marker="o", linewidth=2, label="Linear SVM (Linear Dataset)")
plt.plot(noise_pcts, circ_accuracies, marker="s", linewidth=2, label="RBF SVM (Circular Dataset)")

plt.title("SVM Test Accuracy vs. Training Label Noise")
plt.xlabel("Percentage of Shuffled Training Labels (%)")
plt.ylabel("Clean Test Set Accuracy")
plt.xticks(noise_pcts)
plt.ylim(0.45, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Adjust spacing so labels do not overlap and display the plots
plt.tight_layout()
plt.show()