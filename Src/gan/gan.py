'''
Author: Koen Smallegange 
Goal: Create a GAN that generates option prices
'''

import pandas as pd
import tensorflow as tf
import json
import os
import datetime

from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from networks import make_discriminator_model, make_generator_model, train
from kerastuner.tuners import RandomSearch


# Find the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.json')

# Read in the config file
with open(config_path) as config_file:
    params = json.load(config_file)
    BATCH_SIZE = params['BATCH_SIZE']
    NOISE_DIM = params['NOISE_DIM']
    BUFFER_SIZE = params['BUFFER_SIZE']
    EPOCHS = params['EPOCHS']
    NEURONS_GEN = params['NEURONS_GEN']
    NEURONS_DISC = params['NEURONS_DISC']
    DROPOUT = params['DROPOUT']
    random_samples = params['random_samples']
    accuracy = params['accuracy']
    step_daily = params['step_daily']
    step_weekly = params['step_weekly']
    input_path = params['input_path']
    output_path = params['output_path']

# Create output path
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file_name = f"synthetic_gold_{timestamp}_{EPOCHS}.xlsx"
output_path = f"{output_path}\\Synthetic Data\\{output_file_name}"

# Read and scale data
data = pd.read_excel(input_path)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Instantiate the models, loss function and optimizer
generator = make_generator_model(BATCH_SIZE, NOISE_DIM, NEURONS_GEN, data_scaled)
discriminator = make_discriminator_model(BATCH_SIZE, NEURONS_DISC, DROPOUT, data_scaled)
cross_entropy = tf.keras.losses.BinaryCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

# Initialize the training
train(train_dataset, EPOCHS, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, BATCH_SIZE, NOISE_DIM)

# Generate random noise
random_noise = tf.random.normal([random_samples, NOISE_DIM])

# Use the generator to create option prices
simulated_data = generator(random_noise, training=False)

# Rescale the data and convert to pandas df and save the data
simulated_data_rescaled = scaler.inverse_transform(simulated_data)
simulated_data_df = pd.DataFrame(simulated_data_rescaled, columns=data.columns)
simulated_data_df.to_excel(output_path, index=False)

# Inform user that the program has finished
print(f'Training finished. Synthetic data saved to {output_path}')





