import numpy as np
import matplotlib.pyplot as plt

def is_within_box(x, y, alpha):
    """
    Returns 1 if the point (x, y) is within a box of size alpha centered at the origin,
    otherwise returns 0.
    """
    if -alpha / 2 <= x <= alpha / 2 and -alpha / 2 <= y <= alpha / 2:
        return 1
    return 0

# Create linear mesh grid with noise
x_vals = np.linspace(-5, 5, 30)
y_vals = np.linspace(-3, 3, 30)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

# Add random noise
noise_amplitude = 0.52
x_mesh += np.random.uniform(-noise_amplitude, noise_amplitude, x_mesh.shape)
y_mesh += np.random.uniform(-noise_amplitude, noise_amplitude, y_mesh.shape)

# Compute the grid classification
alpha = 2.0
z_mesh = np.vectorize(is_within_box)(x_mesh, y_mesh, alpha)

# Plot the results
plt.figure(figsize=(6,6))
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        marker = 'X' if z_mesh[j, i] == 1 else 'o'
        color = 'red' if z_mesh[j, i] == 1 else 'blue'
        plt.scatter(x_mesh[j, i], y_mesh[j, i], marker=marker, color=color, s=10)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mesh Grid Classification with Noise')
plt.grid(True)
plt.show()

# Save mesh data
np.savez('mesh_data.npz', x_mesh=x_mesh, y_mesh=y_mesh, z_mesh=z_mesh)


## Example
# Define the size of the box
# Define some example points (x, y)
test_points = np.column_stack((x_vals, y_vals))

# Check if each point is within the box
results = np.zeros(len(test_points))
i = 0
for point in test_points:
    x_test, y_test = point
    results[i] = is_within_box(x_test, y_test, alpha)
    print(f"Point ({x_test}, {y_test}) is within the box: {bool(results[i])}")
    i = i + 1

np.savez('results.npz', test_points=test_points, results=results)
print(f"true results = \n{results}")