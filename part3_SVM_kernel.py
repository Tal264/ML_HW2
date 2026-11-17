# ==========================================
# 3. SVM kernel
# ==========================================

# Import dataset generators from scikit-learn and plotting tools from matplotlib
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

# Import evaluation metrics 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


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
# Initialize the 2 models 
# ==========================================

#  Model 1 (RBF kernel) 
svm_rbf = SVC(kernel="rbf", random_state=random_seed)

#  Model 2 (Polynomial kernel) 
svm_poly = SVC(kernel="poly", degree=3, random_state=random_seed) 


# ==========================================
# Train both models on the circular training data set
# ==========================================

svm_rbf.fit(X_circ_train, y_circ_train)
svm_poly.fit(X_circ_train, y_circ_train)

# ==========================================
# Get predictions and decision scores on X_circ_test
# ==========================================

y_pred_rbf = svm_rbf.predict(X_circ_test) 
y_score_rbf = svm_rbf.decision_function(X_circ_test) 

y_pred_poly = svm_poly.predict(X_circ_test) 
y_score_poly = svm_poly.decision_function(X_circ_test) 


# ==========================================
# Print the three metrics for each model
# ==========================================

# Print metrics for svm_rbf
print("The Score Accuracy:", accuracy_score(y_circ_test, y_pred_rbf))
print("The f1 Score:", f1_score(y_circ_test, y_pred_rbf))
print("The ROC AUC Score:", roc_auc_score(y_circ_test, y_score_rbf))

# Print metrics for svm_poly
print("The Score Accuracy:", accuracy_score(y_circ_test, y_pred_poly))
print("The f1 Score:", f1_score(y_circ_test, y_pred_poly))
print("The ROC AUC Score:", roc_auc_score(y_circ_test, y_score_poly))




# ==========================================
# Visualization of Simulated Datasets
# ==========================================

# Create a figure with 2 row and 2 side-by-side subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))



# Two top rows indicate the dicision boundary display

DecisionBoundaryDisplay.from_estimator(svm_rbf, X_circ_test, response_method="predict", alpha=0.4, ax=axes[0, 0])
axes[0, 0].scatter(X_circ_test[:, 0], X_circ_test[:, 1], c=y_circ_test, s=10, edgecolor="k")
axes[0, 0].set_title(f"RBF Kernel (Acc: {accuracy_score(y_circ_test, y_pred_rbf):.3f})")
axes[0, 0].set_xlabel("X1") 
axes[0, 0].set_ylabel("X2")


DecisionBoundaryDisplay.from_estimator(svm_poly, X_circ_test, response_method="predict", alpha=0.4, ax=axes[0, 1])
axes[0, 1].scatter(X_circ_test[:, 0], X_circ_test[:, 1], c=y_circ_test, s=10, edgecolor="k")
axes[0, 1].set_title(f"Poly Kernel (degree=3) (Acc: {accuracy_score(y_circ_test, y_pred_poly):.3f})")
axes[0, 1].set_xlabel("X1") 
axes[0, 1].set_ylabel("X2")


# Two bottom rows indicate the confusion matrics
ConfusionMatrixDisplay.from_predictions(y_circ_test, y_pred_rbf, ax=axes[1, 0], colorbar=False)
axes[1, 0].set_title("RBF Confusion Matrix")


ConfusionMatrixDisplay.from_predictions(y_circ_test, y_pred_poly, ax=axes[1, 1], colorbar=False)
axes[1, 1].set_title("Poly Confusion Matrix")




# Adjust spacing so labels do not overlap and display the plots
plt.tight_layout()
plt.show()