import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def true_function(x):
    return np.sin(x)

np.random.seed(42)
x_train = np.random.uniform(-2 * np.pi, 2 * np.pi, 1000)
y_train = true_function(x_train) + np.random.normal(0, 0.1, size=len(x_train))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=1000, verbose=0)

x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y_pred = model.predict(x_test)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.plot(x_test, true_function(x_test), color='green', label='True Function')
plt.plot(x_test, y_pred, color='red', label='Predicted Function')
plt.title('Optimizing a Mathematical Function with TensorFlow')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
