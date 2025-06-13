import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Average confusion matrix (rounded from the provided data)
cm_counts = np.array([[274.52, 71.88],
                      [3.38, 2.42]])

# Compute percentages per true label row
row_sums = cm_counts.sum(axis=1, keepdims=True)
cm_percent = cm_counts / row_sums * 100

# Format combined labels
labels = np.array([[f"{cm_counts[i, j]:.2f}\n{cm_percent[i, j]:.2f}%" for j in range(2)] for i in range(2)])

# Plot
plt.figure(figsize=(4, 4))
sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Average confusion matrix Random Forest\nundersampling 67:33 ratio (1=broken)')
plt.tight_layout()
plt.show()
