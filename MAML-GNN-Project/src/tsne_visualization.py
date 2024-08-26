import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Scenario S1 - Example
X1_benign = np.array([
    # Add your 100 benign fixed points here (e.g., [x1, x2, x3, ..., x50])
])

X1_malignant = np.array([
    # Add your 100 malignant fixed points here (e.g., [x1, x2, x3, ..., x50])
])

# Combine benign and malignant data for Scenario S1
X1 = np.vstack([X1_benign, X1_malignant])
y1 = np.array([0] * len(X1_benign) + [1] * len(X1_malignant))

# Scenario S2 - Example
X2_benign = np.array([
    # Add your 100 benign fixed points here
])

X2_malignant = np.array([
    # Add your 100 malignant fixed points here
])

# Combine benign and malignant data for Scenario S2
X2 = np.vstack([X2_benign, X2_malignant])
y2 = np.array([0] * len(X2_benign) + [1] * len(X2_malignant))

# Scenario S3 - Example
X3_benign = np.array([
    # Add your 100 benign fixed points here
])

X3_malignant = np.array([
    # Add your 100 malignant fixed points here
])

# Combine benign and malignant data for Scenario S3
X3 = np.vstack([X3_benign, X3_malignant])
y3 = np.array([0] * len(X3_benign) + [1] * len(X3_malignant))

# Perform t-SNE for Scenario S1
tsne1 = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne1 = tsne1.fit_transform(X1)

# Perform t-SNE for Scenario S2
tsne2 = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne2 = tsne2.fit_transform(X2)

# Perform t-SNE for Scenario S3
tsne3 = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne3 = tsne3.fit_transform(X3)

# Plot for Scenario S1
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne1[y1 == 0, 0], X_tsne1[y1 == 0, 1], color='red', label='Benign')
plt.scatter(X_tsne1[y1 == 1, 0], X_tsne1[y1 == 1, 1], color='blue', label='Malignant')
plt.title('Visualization of Two Classes for Scenario (S1)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Scenario S2
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne2[y2 == 0, 0], X_tsne2[y2 == 0, 1], color='red', label='Benign')
plt.scatter(X_tsne2[y2 == 1, 0], X_tsne2[y2 == 1, 1], color='blue', label='Malignant')
plt.title('Visualization of Two Classes for Scenario (S2)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Scenario S3
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne3[y3 == 0, 0], X_tsne3[y3 == 0, 1], color='red', label='Benign')
plt.scatter(X_tsne3[y3 == 1, 0], X_tsne3[y3 == 1, 1], color='blue', label='Malignant')
plt.title('Visualization of Two Classes for Scenario (S3)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid(True)
plt.show()
