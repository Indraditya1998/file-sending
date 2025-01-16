import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('merged_file.csv')

# Preprocess data
data['lte_rsrp'] = pd.to_numeric(data['lte_rsrp'], errors='coerce')
data['nrg5gsa_rsrp'] = pd.to_numeric(data['nrg5gsa_rsrp'], errors='coerce')
data['nrg5g_nsa_rsrp'] = pd.to_numeric(data['nrg5g_nsa_rsrp'], errors='coerce')
data['lte_rsrq'] = pd.to_numeric(data['lte_rsrq'], errors='coerce')
data['nrg5gsa_rsrq'] = pd.to_numeric(data['nrg5gsa_rsrq'], errors='coerce')
data['nrg5g_nsa_rsrq'] = pd.to_numeric(data['nrg5g_nsa_rsrq'], errors='coerce')

# Fill missing values
data.fillna({
    'lte_rsrp': -140, 'nrg5gsa_rsrp': -140, 'nrg5g_nsa_rsrp': -140,
    'lte_rsrq': -20, 'nrg5gsa_rsrq': -20, 'nrg5g_nsa_rsrq': -20,
    'lat': 0, 'lon': 0, 'speed': 0
}, inplace=True)

# Increase bin granularity
num_bins = 10

# Bin the features
data['lte_rsrp_bin'] = pd.cut(data['lte_rsrp'], bins=num_bins, labels=False)
data['nrg5gsa_rsrp_bin'] = pd.cut(data['nrg5gsa_rsrp'], bins=num_bins, labels=False)
data['nrg5g_nsa_rsrp_bin'] = pd.cut(data['nrg5g_nsa_rsrp'], bins=num_bins, labels=False)
data['lte_rsrq_bin'] = pd.cut(data['lte_rsrq'], bins=num_bins, labels=False)
data['nrg5gsa_rsrq_bin'] = pd.cut(data['nrg5gsa_rsrq'], bins=num_bins, labels=False)
data['nrg5g_nsa_rsrq_bin'] = pd.cut(data['nrg5g_nsa_rsrq'], bins=num_bins, labels=False)
data['lat_bin'] = pd.cut(data['lat'], bins=num_bins, labels=False)
data['lon_bin'] = pd.cut(data['lon'], bins=num_bins, labels=False)
data['speed_bin'] = pd.cut(data['speed'], bins=num_bins, labels=False)

# Define feature set and labels
features = [
    'lat_bin', 'lon_bin', 'speed_bin',
    'lte_rsrp_bin', 'nrg5gsa_rsrp_bin', 'nrg5g_nsa_rsrp_bin',
    'lte_rsrq_bin', 'nrg5gsa_rsrq_bin', 'nrg5g_nsa_rsrq_bin'
]
X = data[features].values
y = data['online'].apply(lambda x: 0 if x == 'rmnet_mhi0.1' else 1).values  # 0 for 5G, 1 for Starlink

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# CNN Model
def create_cnn_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='softmax')  # Softmax for classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
num_actions = 2  # Two actions: 5G or Starlink
input_shape = (X_train_cnn.shape[1], 1)
cnn_model = create_cnn_model(input_shape, num_actions)

# Define reward function using all RSRQ metrics
def calculate_reward(row):
    rsrq_values = [row['lte_rsrq'], row['nrg5gsa_rsrq'], row['nrg5g_nsa_rsrq']]
    avg_rsrq = np.mean(rsrq_values)
    return -avg_rsrq  # Negative because lower RSRQ indicates better signal

data['reward'] = data.apply(calculate_reward, axis=1)

# Training parameters
num_epochs = 20
batch_size = 64

# Train the model
history = cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=num_epochs, batch_size=batch_size, verbose=1)

# Evaluate the model
predictions = cnn_model.predict(X_test_cnn)
y_pred = np.argmax(predictions, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot loss and accuracy curves
plt.figure(figsize=(12, 6))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
cnn_model.save('cnn_dqn_model_final.h5')
