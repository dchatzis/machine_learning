import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# Load the data
data = np.load('mesh_data.npz')

# Extract the mesh arrays
x_mesh = data['x_mesh']
y_mesh = data['y_mesh']
z_mesh = data['z_mesh']

x_flat = x_mesh.flatten()
y_flat = y_mesh.flatten()
z_flat = z_mesh.flatten()

X = np.column_stack((x_flat, y_flat))
Y = z_flat.reshape(-1, 1)

# Print the shapes to verify
print("X", X.shape)
print("Y", Y.shape)
#
print(f"X1 Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"X2 Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"X1 Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"X2 Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

Xt = np.tile(Xn,(500,1))
Yt= np.tile(Y,(500,1))
# #
print(Xt.shape, Yt.shape)
# #

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(4, activation='relu', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
#
model.summary()
#
L1_num_params = 2 * 4 + 4   # W1 parameters  + b1 parameters
L2_num_params = 4 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
#
model.fit(
    Xt,Yt,
    epochs= 10,
)
#
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# #

# Load the test points
data = np.load('results.npz')
test_points = data['test_points']
results = data['results']

test_points_n = norm_l(test_points)
predictions = model.predict(test_points_n)
print("predictions = \n", predictions)
# #
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat.T}")
#yhat = (predictions >= 0.5).astype(int)
#print(f"decisions = \n{yhat.T}")
print(f"true results = \n{results}")