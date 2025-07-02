import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Calculate statistics
mean_val = np.mean(data)
std_val = np.std(data)
median_val = np.median(data)

print(f"Dataset Statistics:")
print(f"Mean: {mean_val:.2f}")
print(f"Standard Deviation: {std_val:.2f}")
print(f"Median: {median_val:.2f}")

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Sample Data Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Histogram generated successfully!")
