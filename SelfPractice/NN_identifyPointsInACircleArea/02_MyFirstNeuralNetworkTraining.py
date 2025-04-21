import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the data
data = np.load('flat_data.npz')
x1_flat = data['x1_flat']
x2_flat = data['x2_flat']
y_flat = data['y_flat']


# Save set into training (70%) and test (30%)
#split the data using sklearn routine
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_flat, x2_flat, y_flat,test_size=0.30, random_state=1)

X_train = np.column_stack((x1_train, x2_train))
Y_train = y_train.reshape(-1, 1)

X_test = np.column_stack((x1_test, x2_test))
Y_test = y_test.reshape(-1, 1)


# Print the shapes to verify
print("X", X_train.shape)
print("Y", Y_train.shape)
#
print(f"X1 Max, Min pre normalization: {np.max(X_train[:,0]):0.2f}, {np.min(X_train[:,0]):0.2f}")
print(f"X2 Max, Min pre normalization: {np.max(X_train[:,1]):0.2f}, {np.min(X_train[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X_train)  # learns mean, variance
Xtrain_n = norm_l(X_train)
print(f"X1 Max, Min post normalization: {np.max(Xtrain_n[:,0]):0.2f}, {np.min(Xtrain_n[:,0]):0.2f}")
print(f"X2 Max, Min post normalization: {np.max(Xtrain_n[:,1]):0.2f}, {np.min(Xtrain_n[:,1]):0.2f}")

Xt = np.tile(Xtrain_n,(500,1))
Yt= np.tile(Y_train,(500,1))
# #
print(Xt.shape, Yt.shape)
# #

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(20, activation='relu', name = 'layer1'),
        Dense(10, activation='relu', name='layer2'),
        Dense(5, activation='relu', name='layer3'),
        Dense(1, activation='linear', name = 'layer4')
     ]
)
#
model.summary()
#

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
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

# Characterize performance
def eval_cat_err(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        if y[i] != yhat[i]:
            incorrect += 1
    cerr = incorrect/m

    return (cerr)

# use the test points to characterize performance
test_points_n = norm_l(X_test)
predictions = model.predict(test_points_n)

y_test_hat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        y_test_hat[i] = 1
    else:
        y_test_hat[i] = 0

caterr = eval_cat_err(Y_test, y_test_hat)
print(caterr)

# Plot misclassified points
incorrect_idx = np.where(y_test_hat != Y_test)[0]
x1_wrong = X_test[incorrect_idx, 0]
x2_wrong = X_test[incorrect_idx, 1]

plt.figure(figsize=(6,6))

# Plot incorrect points
plt.scatter(x1_wrong, x2_wrong, color='red', marker='o', label='Incorrect')

# Plot true decision boundary: circle of radius 5
circle = plt.Circle((0, 0), 5, color='green', fill=False, linewidth=2, label='True Boundary (r=5)')
plt.gca().add_patch(circle)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Misclassified Points with True Decision Boundary')
plt.grid(True)
plt.axis('equal')  # Ensures the circle is actually circular
plt.legend()
plt.show()