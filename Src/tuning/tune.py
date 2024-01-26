import pandas as pd
import tensorflow as tf
import json
import os
import datetime
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch

# Import your make_discriminator_model, make_generator_model, and train functions from 'networks' module
from networks import make_discriminator_model, make_generator_model, train

# Function to define your GAN model creation and training logic
def train_gan(hp):
    BATCH_SIZE = hp.Int('batch_size', min_value=32, max_value=256, step=32)
    NOISE_DIM = hp.Int('noise_dim', min_value=32, max_value=256, step=32)
    NEURONS_GEN = hp.Int('neurons_gen', min_value=64, max_value=512, step=64)
    NEURONS_DISC = hp.Int('neurons_disc', min_value=64, max_value=512, step=64)
    DROPOUT = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)

    # Load your data (data_scaled) and preprocess it as needed
    data = pd.read_excel(r"C:\\Users\\koens\\OneDrive\\Bureaublad\\Research-Practicum\\Data\\Real Data\\gold_preprocessed.xlsx")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    train_dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Create generator and discriminator models based on hyperparameters
    generator = make_generator_model(BATCH_SIZE, NOISE_DIM, NEURONS_GEN, data_scaled)
    discriminator = make_discriminator_model(BATCH_SIZE, NEURONS_DISC, DROPOUT, data_scaled)

    # Define loss function and optimizers
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()

    # Compile the generator model
    generator.compile(optimizer=generator_optimizer, loss=cross_entropy)

    # Initialize training
    train(train_dataset, EPOCHS, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, BATCH_SIZE, NOISE_DIM)

    # Return the generator model
    return generator

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

# Create a Keras Tuner RandomSearch tuner
tuner = RandomSearch(
    train_gan,  # Pass the train_gan function
    objective='loss',  # Specify the metric to optimize (e.g., GAN loss)
    max_trials=5,  # Number of trials for hyperparameter search
    directory='my_tuner_directory',  # Directory to store tuner results
    project_name='my_gan_tuning')  # Project name for the tuner

# Start the hyperparameter search
tuner.search(epochs=EPOCHS, verbose=1)

# Get the best hyperparameters and create the final GAN model
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_gan_model = train_gan(best_hyperparameters)

print(best_hyperparameters.values)