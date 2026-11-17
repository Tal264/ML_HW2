# ==========================================
# # 6. High dimensionality
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

# ==========================================
# Helper function - label-shuffling function & loop over noise percentages
# ==========================================
dims = [2, 5, 10, 15]
lin_clean_accs, lin_noisy_accs = [], []
circ_clean_accs, circ_noisy_accs = [], []

def shuffle_labels(y, pct, seed):
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_shuffle = int(pct * len(y))
    idx = rng.choice(len(y), size=n_shuffle, replace=False)
    y_noisy[idx] = rng.permutation(y_noisy[idx])
    return y_noisy

for d in dims:
    # 1. Generate d-dimensional Linear dataset
    X_lin_d, y_lin_d = make_blobs(
        n_samples=n, n_features=d, centers=[(-5,) * d, (5,) * d], cluster_std=1.0, random_state=random_seed
    )

    # 2. Generate d-dimensional Concentric Hyperspheres
    rng = np.random.default_rng(random_seed)
    y_circ_d = np.array([0] * (n // 2) + [1] * (n // 2))
    dirs = rng.normal(size=(n, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = np.where(y_circ_d == 0, 0.5, 1.0)[:, None]
    X_circ_d = dirs * radii + rng.normal(scale=0.12, size=(n, d))

    # 3. Train/test split for both datasets
    X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(
        X_lin_d, y_lin_d, test_size=0.2, random_state=random_seed
    )
    X_circ_train, X_circ_test, y_circ_train, y_circ_test = train_test_split(
        X_circ_d, y_circ_d, test_size=0.2, random_state=random_seed
    )

    # 4. Create 50% shuffled noisy training labels
    y_lin_noisy = shuffle_labels(y_lin_train, 0.50, random_seed)
    y_circ_noisy = shuffle_labels(y_circ_train, 0.50, random_seed)

    # 5. Linear SVM (Clean vs. 50% Shuffled Noise)
    svm_lin_clean = SVC(kernel="linear", random_state=random_seed).fit(X_lin_train, y_lin_train)
    svm_lin_noisy = SVC(kernel="linear", random_state=random_seed).fit(X_lin_train, y_lin_noisy)
    lin_clean_accs.append(svm_lin_clean.score(X_lin_test, y_lin_test))
    lin_noisy_accs.append(svm_lin_noisy.score(X_lin_test, y_lin_test))

    # 6. RBF SVM (Clean vs. 50% Shuffled Noise)
    svm_rbf_clean = SVC(kernel="rbf", random_state=random_seed).fit(X_circ_train, y_circ_train)
    svm_rbf_noisy = SVC(kernel="rbf", random_state=random_seed).fit(X_circ_train, y_circ_noisy)
    circ_clean_accs.append(svm_rbf_clean.score(X_circ_test, y_circ_test))
    circ_noisy_accs.append(svm_rbf_noisy.score(X_circ_test, y_circ_test))

    print(
        f"Dim {d:2d} -> Linear (Clean/Noisy): {lin_clean_accs[-1]:.4f} / {lin_noisy_accs[-1]:.4f} | "
        f"RBF Circle (Clean/Noisy): {circ_clean_accs[-1]:.4f} / {circ_noisy_accs[-1]:.4f}"
    )

# ==========================================
# Plot performance across dimensions
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(dims, lin_clean_accs, marker="o", linewidth=2, label="Linear SVM (Clean)")
plt.plot(dims, lin_noisy_accs, marker="o", linestyle="--", linewidth=2, label="Linear SVM (50% Shuffled Noise)")
plt.plot(dims, circ_clean_accs, marker="s", linewidth=2, label="RBF SVM (Clean)")
plt.plot(dims, circ_noisy_accs, marker="s", linestyle="--", linewidth=2, label="RBF SVM (50% Shuffled Noise)")

plt.title("SVM Test Accuracy vs. Dimensionality (Clean vs. Noisy Labels)")
plt.xlabel("Number of Dimensions (d)")
plt.ylabel("Clean Test Set Accuracy")
plt.xticks(dims)
plt.ylim(0.45, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()