# ==========================================
# 2. SVM and Perceptron Comparison
# ==========================================

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Define total number of samples (n=10,000) and a fixed random seed for reproducibility
n = 10000
random_seed = 40

# Generate the Linear dataset:
# Two 2D Gaussian clusters centered far apart at (-5, -5) and (5, 5) with std=1.0
X_lin, y_lin = make_blobs(
    n_samples=n, centers=[(-5, -5), (5, 5)], cluster_std=1.0, random_state=random_seed
)

# Generate the Circular dataset:
# Two concentric rings (inner radius = 0.5 * outer radius) with slight Gaussian noise
X_circ, y_circ = make_circles(
    n_samples=n, factor=0.5, noise=0.12, random_state=random_seed
)

# ==========================================
# Split both datasets into training and testing sets (80% train / 20% test)
# ==========================================
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=random_seed
)
X_circ_train, X_circ_test, y_circ_train, y_circ_test = train_test_split(
    X_circ, y_circ, test_size=0.2, random_state=random_seed
)

# ==========================================
# Initialize and train the 4 models
# ==========================================
# Perceptron on Linear
perc_lin = Perceptron(random_state=random_seed)
perc_lin.fit(X_lin_train, y_lin_train)

# Linear SVM on Linear
svm_lin = SVC(kernel="linear", random_state=random_seed)
svm_lin.fit(X_lin_train, y_lin_train)

# Perceptron on Circular
perc_circ = Perceptron(random_state=random_seed)
perc_circ.fit(X_circ_train, y_circ_train)

# Linear SVM on Circular
svm_circ = SVC(kernel="linear", random_state=random_seed)
svm_circ.fit(X_circ_train, y_circ_train)

# Print test accuracies
print(f"Perceptron Linear Accuracy:   {perc_lin.score(X_lin_test, y_lin_test):.4f}")
print(f"Linear SVM Linear Accuracy:   {svm_lin.score(X_lin_test, y_lin_test):.4f}")
print(f"Perceptron Circular Accuracy: {perc_circ.score(X_circ_test, y_circ_test):.4f}")
print(f"Linear SVM Circular Accuracy: {svm_circ.score(X_circ_test, y_circ_test):.4f}")

# ==========================================
# Visualization of Decision Boundaries & Margins
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

models_info = [
    (axes[0, 0], perc_lin, X_lin_test, y_lin_test, "Perceptron - Linear"),
    (axes[0, 1], svm_lin, X_lin_test, y_lin_test, "Linear SVM - Linear"),
    (axes[1, 0], perc_circ, X_circ_test, y_circ_test, "Perceptron - Circular"),
    (axes[1, 1], svm_circ, X_circ_test, y_circ_test, "Linear SVM - Circular"),
]

for ax, model, X_test, y_test, title in models_info:
    # Shaded decision regions
    DecisionBoundaryDisplay.from_estimator(
        model, X_test, response_method="predict", alpha=0.4, ax=ax
    )

    # Plot decision boundary (solid) and margins (dashed) for Linear SVM
    if isinstance(model, SVC):
        DecisionBoundaryDisplay.from_estimator(
            model,
            X_test,
            response_method="decision_function",
            plot_method="contour",
            levels=[-1, 0, 1],
            linestyles=["--", "-", "--"],
            colors="k",
            ax=ax,
        )

    # Scatter plot of test points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=12, edgecolor="k", linewidth=0.4)
    ax.set_title(f"{title} (Acc: {model.score(X_test, y_test):.3f})")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

plt.tight_layout()
plt.show()