'''
This file contains the functions that define the networks used in the GAN.
'''

import tensorflow as tf
from tensorflow.keras import layers, Model


def make_generator_model(BATCH_SIZE, NOISE_DIM, NEURONS_GEN, data_scaled):
    '''Generates a network that generates synthetic data'''
    generator = tf.keras.Sequential([
        layers.Dense(BATCH_SIZE, activation='relu', input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.Dense(NEURONS_GEN, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(data_scaled.shape[1], activation='tanh')
    ])
    return generator

def make_discriminator_model(BATCH_SIZE, NEURONS_DISC, DROPOUT, data_scaled):
    '''Generates a network that discriminates real from synthetic data'''
    discriminator = tf.keras.Sequential([
        layers.Dense(BATCH_SIZE, activation='relu', input_shape=(data_scaled.shape[1],)),
        layers.Dropout(DROPOUT),
        layers.Dense(NEURONS_DISC, activation='relu'),
        layers.Dropout(DROPOUT),
        layers.Dense(1, activation='sigmoid')
    ])
    return discriminator


@tf.function
def train_step(real_data, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, BATCH_SIZE, NOISE_DIM):
    '''Trains the GAN for one step'''	

    # Sample noise from a normal distribution
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # Train the discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        
        # Discriminate real from fake data
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)
        
        # Calculate the losses
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) +
                     cross_entropy(tf.zeros_like(fake_output), fake_output))
        
    # Calculate the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Update the weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Train the model in a certain number of epochs
def train(dataset, epochs, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, BATCH_SIZE, NOISE_DIM):
    '''Trains the GAN'''
    for epoch in range(epochs):
        for data_batch in dataset:
            train_step(data_batch, generator, discriminator, cross_entropy, generator_optimizer, discriminator_optimizer, BATCH_SIZE, NOISE_DIM)

        # Display progress and remove previous output
        print(f'Training status: {round(epoch/epochs*100, 2)} % of {epochs} epochs completed', end='\r')
   