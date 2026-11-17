# Machine Learning and Neural Networks, HW 2, Bar-Ilan University
## 2HW – Machine Learning & Neural Networks for Neuroscience

### Instructions
* **a.** The deadline for submission is November 16th at 23:59 to ''.
* **b.** Submit in pairs.
* **c.** The submission email should include:
  * **i)** the final PDF file
  * **ii)** the Python scripts you used to generate the main figures and analysis (preferably in GitHub).

---

### 1. Dataset simulation (15 points)
Simulate two datasets (use a fixed random seed for reproducibility). The inputs should be 2-dimensional, namely, two vectors, `X1` and `X2`, each of length `n` (e.g., `n = 10,000`). The output should be a binary class `Y` according to the following assumptions:
* **a. Linear dataset:** the two clusters of dots should not overlap. Rather, be far from each other (Figure a).
* **b. Circular dataset:** the two clusters of dots should overlap, i.e., one cluster is in a circle inside the other cluster (Figure b).

Plot the two datasets on a scatter plot. They should look similar to the basic example below (note: you do not need to simulate this exact dataset, but a very similar one!).

---

### 2. SVM and perceptron comparison (20 points)
Compare the performance of a simple perceptron and a linear SVM for both datasets. Add plots and visualize your results to support your claims.

---

### 3. SVM kernel (15 points)
Add a kernel to the SVM to separate the circular dataset.
Compare two possible kernels and show which is better using plots and relevant evaluation matrices (e.g., confusion matrix, accuracy, F1-score, or AUC).

---

### 4. Data augmentation (20 points)
Examine the overfitting of the best-fitted SVM model by data augmentation. For that, implement the following simple augmentation:
* **a.** Sample 1000 datapoints as a new dataset
* **b.** Add a Gaussian noise to the sampled dataset (one of two options: low or high SD; see below).
* **c.** Repeat a-b for `k` times (`k >> 1000`).

This means that in each augmented dataset, the dots you create will move in a random direction (pulled from a Gaussian distribution).
Visualize and explain your generalization results by using relevant evaluation matrices to demonstrate the different augmentations in scatter plots.

---

### 5. Noise analysis (20 points)
How does misclassification affect the performance of your SVM?
In real-life data, the labels of data points do not always reflect a real class/label or can be noisy (e.g., a naïve annotation mistake). One option to test is to shuffle some percentage of the labels in the training set (not on the augmented data, only on the original datasets) and then rerun the classifier.
Plot a graph that shows the performance as a function of the percentage of shuffled labels (use three options, e.g., 15%, 25%, 50%), and write your conclusion.

---

### 6. High dimensionality (10 points)
How do higher dimensions affect your answers to the previous section?
* **a.** To test that, create new datasets of higher dimensions (e.g., 5, 10, and 15 instead of 2).

Plot the performance of the SVM as a function of three possible higher dimensions.

---

### Bonus (5 points)
To compare the effect of overfitting, choose a threshold of performance (for example, 70%) and find the point where the SVM performance is decreased for each augmentation.