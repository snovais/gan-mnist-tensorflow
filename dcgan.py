# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_docs.vis.embed as embed

import glob

import imageio

import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

import time

from IPython import display

# carregar e preparar dados
# necessário para a rede discriminadora

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # normaliza as imagens entre [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Mini lote e embalhamento
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# cria os modelos
""" O gerador usa camadas `tf.keras.layers.Conv2DTranspose` 
    (upsampling) para produzir uma imagem a partir de uma 
    semente (ruído aleatório). Comece com uma camada `Densa`
    que recebe esta semente como entrada e, em seguida, 
    faça upsample várias vezes até atingir o tamanho de imagem
    desejado de 28x28x1. Observe a ativação 
    `tf.keras.layers.LeakyReLU` para cada camada, 
    exceto a camada de saída que usa tanh.
"""

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

#Use o gerador (ainda não treinado) para criar uma imagem.
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# Rede discriminadora
# baseada em uma rede neural convolucional para classificacao de imagens


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

""" Use o discriminador (ainda não treinado) para 
    classificar as imagens geradas como reais ou falsas.
    O modelo será treinado para gerar valores positivos para
    imagens reais e valores negativos para imagens falsas."""
 
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# Define a função de perda e o otimizador para os dois modelos

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Perda, este método quantifica o quão bem o discriminador é capaz de distinguir
# entre as imagens reais das falsificações

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define a função de perda e quantifica quão bem consegue enganar o discriminador

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# O discriminador e os otimizadores do gerador são diferentes, pois estará treinando
# duas redes separadamente.
generator_optimizer = tf.keras.optimizers.Adamax(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adamax(1e-3)

""" Salva pontos durante o processamento
    e também torna possível restaurar caso algum problema
"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Define o loop de treinamento

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16


seed = tf.random.normal([num_examples_to_generate, noise_dim])

# o treinamento começa recebendo ruído como entrada

# decorator transforma a funcao em grafo para acelerar o processamento
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

# Gerando e salvando as imagens

def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Chame o método `train ()` definido acima para treinar o gerador e o discriminador
# simultaneamente. Observe que treinar GANs pode ser complicado. É importante que o 
# gerador e o discriminador não se sobreponham (por exemplo, que treinem a uma taxa 
# semelhante).

# No início do treinamento, as imagens geradas parecem ruídos aleatórios. Conforme o treinamento avança, os dígitos gerados parecerão cada vez mais reais. Após cerca de 50 épocas, eles se assemelham aos dígitos MNIST. Isso pode levar cerca de um minuto / período com as configurações padrão do Colab.


train(train_dataset, EPOCHS)

# Restaure o último ponto de parada

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# cria gif

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

"""Use `imageio` to create an animated gif using the images saved during training."""

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


embed.embed_file(anim_file)
