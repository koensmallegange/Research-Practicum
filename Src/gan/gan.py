'''
Author: Koen Smallegange 
Goal: Create a GAN that generates option prices
'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from networks import make_discriminator_model, make_generator_model, train
import json

# Read data
data = pd.read_excel(r"C:\Users\koens\OneDrive\Bureaublad\Research-Practicum\Data\Real Data\gold_preprocessed.xlsx")

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)

# Get hyperparameters from json file
with open(r"C:\Users\koens\OneDrive\Bureaublad\Research-Practicum\SRC\gan\config.json") as config_file:
    params = json.load(config_file)
    BATCH_SIZE = params['BATCH_SIZE']
    NOISE_DIM = params['NOISE_DIM']
    BUFFER_SIZE = params['BUFFER_SIZE']
    EPOCHS = params['EPOCHS']
    random_samples = params['random_samples']
    accuracy = params['accuracy']
    step_daily = params['step_daily']
    step_weekly = params['step_weekly']


# Set hyperparameters
# BATCH_SIZE = 128
# NOISE_DIM = 75
# BUFFER_SIZE = 5389 
# EPOCHS = 7500
# random_samples = 5211
# accuracy = 5
# step_daily = 0.00273785078
# step_weekly = 0.14285714285

# Define the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Instantiate the models
generator = make_generator_model(BATCH_SIZE, NOISE_DIM, data_scaled)
discriminator = make_discriminator_model(BATCH_SIZE, data_scaled)

# Define the loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

# Initialize the training
train(train_dataset, EPOCHS, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, BATCH_SIZE, NOISE_DIM)

# Generate random noise
random_noise = tf.random.normal([random_samples, NOISE_DIM])

# Use the generator to create option prices
simulated_data = generator(random_noise, training=False)

# Rescale the data
simulated_data_rescaled = scaler.inverse_transform(simulated_data)

# Convert the generated data to pandas df
simulated_data_df = pd.DataFrame(simulated_data_rescaled, columns=data.columns)

# Save synthetic data to drive
simulated_data_df.to_excel(r"C:\Users\koens\OneDrive\Bureaublad\Research-Practicum\Data\Synthetic Data\synthetic_gold_lowsamples_7500.xlsx", index=False)





