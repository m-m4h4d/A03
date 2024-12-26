# Enhanced Soft Clustering Algorithm

This repository contains the implementation of an Enhanced Soft Clustering Algorithm, which is designed to address the limitations of traditional soft clustering methods, such as Fuzzy C-Means (FCM). The algorithm incorporates a hybrid distance measure (combining Euclidean and Mahalanobis distances) to improve clustering performance in datasets with diverse distributions.

## Features

- Hybrid distance measure for improved robustness.
- Adjustable fuzziness parameter (`m`) and hybridization coefficient (`alpha`).
- Convergence monitoring with tolerance-based stopping criteria.
- Suitable for synthetic and real-world datasets.

## Installation

To run the implementation, ensure you have Python installed along with the required libraries:

```bash
pip install numpy scipy matplotlib scikit-learn
```

## Usage

1. **Generate Synthetic Data**:
   The implementation uses the `make_blobs` function from `sklearn.datasets` to generate synthetic data for testing.

2. **Initialize the Algorithm**:
   ```python
   clustering = EnhancedSoftClustering(n_clusters=4, alpha=0.5)
   ```

3. **Fit the Model**:
   ```python
   clustering.fit(X)
   ```

4. **Visualize Results**:
   ```python
   import matplotlib.pyplot as plt

   plt.scatter(X[:, 0], X[:, 1], c=np.argmax(clustering.membership, axis=1), cmap='viridis')
   plt.scatter(clustering.centers[:, 0], clustering.centers[:, 1], c='red', marker='x')
   plt.title("Enhanced Soft Clustering")
   plt.show()
   ```

## Parameters

- **`n_clusters`**: Number of clusters to form.
- **`m`**: Fuzziness parameter. Values >1 result in soft clustering.
- **`alpha`**: Hybridization coefficient (0: Mahalanobis distance, 1: Euclidean distance).
- **`max_iter`**: Maximum number of iterations.
- **`tol`**: Convergence tolerance for center updates.

## Example

The following example demonstrates clustering on a 2D synthetic dataset with 4 clusters:

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
clustering = EnhancedSoftClustering(n_clusters=4, alpha=0.5)
clustering.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=np.argmax(clustering.membership, axis=1), cmap='viridis')
plt.scatter(clustering.centers[:, 0], clustering.centers[:, 1], c='red', marker='x')
plt.title("Enhanced Soft Clustering")
plt.show()
```

## Performance Metrics

The algorithm can be evaluated using the following metrics:

- **Clustering Accuracy**
- **Silhouette Score**
- **Execution Time**

## License

This project is licensed under the MIT License. Feel free to use and modify the code for academic and research purposes.
