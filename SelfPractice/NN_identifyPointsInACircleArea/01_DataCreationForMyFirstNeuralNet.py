import numpy as np
import matplotlib.pyplot as plt


def is_within_circle(x1, x2, r):
    """
    Returns 1 if the point (x1, x2) is within a circle of radius r centered at the origin,
    otherwise returns 0.
    """
    if x1**2 + x2**2 <= r**2:
        return 1
    return 0


# Create linear mesh grid with noise
x1_vals = np.linspace(-10, 10, 100)
x2_vals = np.linspace(-10, 10, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)

# Add random noise
noise_amplitude = 1.15
x1_mesh += np.random.uniform(-noise_amplitude, noise_amplitude, x1_mesh.shape)
x2_mesh += np.random.uniform(-noise_amplitude, noise_amplitude, x2_mesh.shape)

# Compute the grid classification
alpha = 5.0
y_mesh = np.vectorize(is_within_circle)(x1_mesh, x2_mesh, alpha)

# Plot the results
plt.figure(figsize=(6, 6))
for i in range(len(x1_vals)):
    for j in range(len(x2_vals)):
        marker = 'X' if y_mesh[j, i] == 1 else 'o'
        color = 'red' if y_mesh[j, i] == 1 else 'blue'
        plt.scatter(x1_mesh[j, i], x2_mesh[j, i], marker=marker, color=color, s=10)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Mesh Grid Classification with Noise')
plt.grid(True)
plt.show()

# Before saving, shuffle the data order
# Flatten the mesh grids and labels
x1_flat = x1_mesh.flatten()
x2_flat = x2_mesh.flatten()
y_flat = y_mesh.flatten()

# Generate a random permutation of indices
perm = np.random.permutation(len(x1_flat))

# Shuffle all arrays using the same permutation
x1_shuffled = x1_flat[perm]
x2_shuffled = x2_flat[perm]
y_shuffled = y_flat[perm]

# Reshape back to original mesh shape
x1_mesh_shuffled = x1_shuffled.reshape(x1_mesh.shape)
x2_mesh_shuffled = x2_shuffled.reshape(x2_mesh.shape)
y_mesh_shuffled = y_shuffled.reshape(y_mesh.shape)

# Optional: plot the reshuffled mesh for verification
plt.figure(figsize=(6, 6))
for i in range(len(x1_vals)):
    for j in range(len(x2_vals)):
        marker = 'X' if y_mesh_shuffled[j, i] == 1 else 'o'
        color = 'red' if y_mesh_shuffled[j, i] == 1 else 'blue'
        plt.scatter(
            x1_mesh_shuffled[j, i], x2_mesh_shuffled[j, i], marker=marker, color=color, s=10)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Shuffled Mesh Grid Classification')
plt.grid(True)
plt.show()

# Save full data set
np.savez(
    'mesh_data.npz', x1_mesh=x1_mesh_shuffled, x2_mesh=x2_mesh_shuffled, y_mesh=y_mesh_shuffled)
np.savez('flat_data.npz', x1_flat=x1_shuffled, x2_flat=x2_shuffled, y_flat=y_shuffled)
