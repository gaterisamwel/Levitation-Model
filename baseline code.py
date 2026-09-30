import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def generate_trajectory(samples=1000, timesteps=50):
    X, Y = [], []
    for _ in range(samples):
        mass = np.random.uniform(1, 10)
        thrust_power = np.random.uniform(5, 20)
        B = np.random.uniform(0.1, 2.0)
        distance = np.random.uniform(0.5, 5.0)
        density = np.random.uniform(0.8, 1.2)

        coords = []
        altitude = np.random.uniform(0, 2)
        x, y = 0.0, 0.0

        for t in range(timesteps):
            Fg = mass * 9.81
            Ft = thrust_power / mass
            Fm = (B**2) / (distance**2)

            a_z = (Ft + Fm - Fg) / mass
            a_z += np.random.normal(0, 0.05)  

            altitude += a_z * 0.1
            altitude = max(0, altitude)

            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)

            coords.append([x, y, altitude])

        features = [mass, thrust_power, B, distance, density]
        X.append(np.tile(features, (timesteps, 1)))
        Y.append(coords)

    return np.array(X), np.array(Y)
X, Y = generate_trajectory(samples=1000, timesteps=50)
Y_altitude = Y[:, -1, 2]  
X_flat = X[:, 0, :]        
X_train, X_test, y_train, y_test = train_test_split(X_flat, Y_altitude, test_size=0.2)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.3f}")
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
