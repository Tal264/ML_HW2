# ==========================================
# 4. Data Augmentation (Generalization & Overfitting Check)
# ==========================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Reproduce the Circular dataset and train the best model from Q3 (RBF SVM)
n = 10000
random_seed = 40
rng = np.random.default_rng(random_seed)

X_circ, y_circ = make_circles(
    n_samples=n, factor=0.5, noise=0.12, random_state=random_seed
)
X_circ_train, X_circ_test, y_circ_train, y_circ_test = train_test_split(
    X_circ, y_circ, test_size=0.2, random_state=random_seed
)

best_svm = SVC(kernel="rbf", random_state=random_seed)
best_svm.fit(X_circ_train, y_circ_train)

# 2. Data Augmentation Loop (Steps a, b, c)
k = 1500            # Number of repetitions (k > 1000)
sample_size = 1000  # Sample 1,000 datapoints per iteration
low_sd = 0.05       # Low Gaussian noise SD
high_sd = 0.35      # High Gaussian noise SD

results = {
    "Low SD": {"acc": [], "f1": [], "auc": []},
    "High SD": {"acc": [], "f1": [], "auc": []},
}

for label, sd in [("Low SD", low_sd), ("High SD", high_sd)]:
    for _ in range(k):
        # Step a: Randomly sample 1,000 datapoints from the test set
        idx = rng.choice(len(X_circ_test), size=sample_size, replace=False)
        X_sample = X_circ_test[idx]
        y_sample = y_circ_test[idx]

        # Step b: Add 2D Gaussian noise in random directions
        noise = rng.normal(loc=0.0, scale=sd, size=X_sample.shape)
        X_aug = X_sample + noise

        # Evaluate the pre-trained RBF SVM without retraining
        y_pred = best_svm.predict(X_aug)
        y_score = best_svm.decision_function(X_aug)

        results[label]["acc"].append(accuracy_score(y_sample, y_pred))
        results[label]["f1"].append(f1_score(y_sample, y_pred))
        results[label]["auc"].append(roc_auc_score(y_sample, y_score))

# Print summary statistics (Mean ± SD across k=1,500 augmented datasets)
for label in ["Low SD", "High SD"]:
    print(f"=== {label} (SD = {low_sd if label == 'Low SD' else high_sd}, k = {k}) ===")
    print(f"Accuracy: {np.mean(results[label]['acc']):.4f} ± {np.std(results[label]['acc']):.4f}")
    print(f"F1-Score: {np.mean(results[label]['f1']):.4f} ± {np.std(results[label]['f1']):.4f}")
    print(f"ROC-AUC:  {np.mean(results[label]['auc']):.4f} ± {np.std(results[label]['auc']):.4f}\n")

# 3. Visualization: Scatter Plots + Evaluation Metric Distributions
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Generate one representative 1,000-point sample for each noise level to visualize
sample_idx = rng.choice(len(X_circ_test), size=sample_size, replace=False)
X_base, y_base = X_circ_test[sample_idx], y_circ_test[sample_idx]
X_aug_low = X_base + rng.normal(0.0, low_sd, size=X_base.shape)
X_aug_high = X_base + rng.normal(0.0, high_sd, size=X_base.shape)

# Top-Left: Scatter plot of Low SD augmented data with RBF boundary
DecisionBoundaryDisplay.from_estimator(
    best_svm, X_aug_high, response_method="predict", alpha=0.35, ax=axes[0, 0]
)
axes[0, 0].scatter(X_aug_low[:, 0], X_aug_low[:, 1], c=y_base, s=12, edgecolor="k", linewidth=0.3)
axes[0, 0].set_title(f"Low SD Augmentation (SD={low_sd}) — Acc: {accuracy_score(y_base, best_svm.predict(X_aug_low)):.3f}")
axes[0, 0].set_xlabel("X1")
axes[0, 0].set_ylabel("X2")

# Top-Right: Scatter plot of High SD augmented data with RBF boundary
DecisionBoundaryDisplay.from_estimator(
    best_svm, X_aug_high, response_method="predict", alpha=0.35, ax=axes[0, 1]
)
axes[0, 1].scatter(X_aug_high[:, 0], X_aug_high[:, 1], c=y_base, s=12, edgecolor="k", linewidth=0.3)
axes[0, 1].set_title(f"High SD Augmentation (SD={high_sd}) — Acc: {accuracy_score(y_base, best_svm.predict(X_aug_high)):.3f}")
axes[0, 1].set_xlabel("X1")
axes[0, 1].set_ylabel("X2")

# Bottom-Left: Histogram of Accuracy across k=1,500 runs
axes[1, 0].hist(results["Low SD"]["acc"], bins=30, alpha=0.7, label=f"Low SD (Mean={np.mean(results['Low SD']['acc']):.3f})")
axes[1, 0].hist(results["High SD"]["acc"], bins=30, alpha=0.7, label=f"High SD (Mean={np.mean(results['High SD']['acc']):.3f})")
axes[1, 0].set_title(f"Accuracy Distribution Across k={k} Augmented Datasets")
axes[1, 0].set_xlabel("Accuracy")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()

# Bottom-Right: Scatter plot of F1-Score vs ROC-AUC across k=1,500 runs
axes[1, 1].scatter(results["Low SD"]["auc"], results["Low SD"]["f1"], alpha=0.35, s=12, label="Low SD")
axes[1, 1].scatter(results["High SD"]["auc"], results["High SD"]["f1"], alpha=0.35, s=12, label="High SD")
axes[1, 1].set_title(f"F1-Score vs. ROC-AUC Across k={k} Runs")
axes[1, 1].set_xlabel("ROC-AUC")
axes[1, 1].set_ylabel("F1-Score")
axes[1, 1].legend()

plt.tight_layout()
plt.show()