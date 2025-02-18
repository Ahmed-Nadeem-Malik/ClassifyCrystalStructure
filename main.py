import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# set a seed so results are always the same reprodicibility!
np.random.seed(42)

# Generate some fake data for teh project
num_points = 500
lat_size = np.random.uniform(3.0, 5.0, num_points)      # random lattice sizes between 3.0 and 5.0
bond_len = np.random.uniform(1.5, 2.5, num_points)        # random bond lengths between 1.5 and 2.5
symm_val = np.random.uniform(0.1, 1.0, num_points)        # random symmetry values between 0.1 and 1.0

# Create a "score" to decide our classes this is a weighted sum of our features
#Arbitary scoring system taht is just taken from genrative ai just because I dont have a formula yet may iclude one in the future
score = 0.5 * lat_size + 0.3 * bond_len + 0.2 * symm_val

# Normalize the score so it falls between 0 and 1 (min-max normalization)
norm_score = (score - score.min()) / (score.max() - score.min())

# Now assign labels based on the normalized score:
# If norm_score <= 0.33 -> label 0, between 0.33 and 0.66 -> label 1, and > 0.66 -> label 2
#Diffrent labels pressent diffrent crystal structures
labels = np.zeros(num_points, dtype=int)
labels[norm_score > 0.33] = 1
labels[norm_score > 0.66] = 2

# Create a DataFrame to hold our data makes things more organized
#Three values that are used to measure how a crystal is classiffied, there is a lot more irl so note to self maybe include it in the future
df = pd.DataFrame({
    'LatSize': lat_size,
    'BondLen': bond_len,
    'SymmVal': symm_val,
    'Class': labels
})

# Split the data into features and target variable
features = df.drop(columns=['Class'])
target = df['Class']

# Now split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features so they're on a similar scale (helps the training process)
#allows a fair comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build our neural network model
#filters it through the neurfal network
model = Sequential([
    Input(shape=(3,)),                # we have 3 features
    Dense(32, activation='relu'),     # first hidden layer with 32 neurons
    Dropout(0.2),                     # dropout to avoid overfitting (drop some neurons randomly)
    Dense(16, activation='relu'),     # second hidden layer with 16 neurons
    Dense(3, activation='softmax')     # output layer: 3 classes
])

# Compile the model using adam optimizer and proper loss function for multiclass classification
#
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Setup early stopping to halt training if the validation loss doesnt improve after a while
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model - it will print out epoch progress
history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# Evaluate the model on the test data and print the accuracy
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Acc: {test_acc:.2f}")

# Plot train vs. validation accuracy over the epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accurcy')  
plt.title('Train vs Val Acc')
plt.legend()
plt.show()

# Predict on a new datapoint
new_datapoint = np.array([[4.2, 2.0, 0.5]])  # example: [lattice size, bond length, symmetry value]
new_datapoint_scaled = scaler.transform(new_datapoint)
pred = model.predict(new_datapoint_scaled)
predicted_class = np.argmax(pred)
print(f"Predicted Crystal Structure: {predicted_class}")
