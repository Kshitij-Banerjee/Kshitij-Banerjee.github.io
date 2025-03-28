---
Category: ML Trading
Title: Auto-encoders, VAE and GAN - Part 3 
Layout: post
Name: Auto-encoders, VAE, and GAN - Part 3
banner: "VAE_Banner.png"
draft: True
cover:
  image: "VAE_Banner.png"
date: 2025-03-23
tags: [ML, AI]
keywords: [ML, AI, Trading, AutoEncoders]
---

# Introduction

In my previous [post](https://kshitijbanerjee.com/2024/12/25/adventures-in-ml-trading-part-2/), I discussed this idea of understanding the underlying state in which the stock is (also called regime detection), and using it to enhance our probabilistic model. 

However, I used a very rudimentary way to establish the current state (i.e, look at past 3 day returns and sample from prior instances with similar 3-day-return). 
I now want to improve on that idea, and establish the current regime based on an ML based strategy. But, to understand how to do this, we must understand some core concepts. 

In this post, I'll cover some theory around

1. Autoencoders
2. Variational Auto Encoders
3. GAN


# Autoencoder

In essence, we predict a certain embedding vector from the encoder that represents the input in a (usually compressed) latent space.
![EncoderDecoder](/encoder_decoder.png)
We then use this vector representation, and decode the original input from this embedding. 
When we train such a system end to end, the model is forced to find optimal embedding vectors that capture the essence of the input.


# Variational Auto Encoder

In Autoencoders, we converted the input into a specific vector point in the embedding space and aimed to recontruct the exact same input from this embedding. 

In variational auto encoders, we instead ask the encoder to output a _probability distribution_ that represents the input. 
The encoder now outputs a mean, and variance over the output distribution instead of a single point. 
This allows us to sample from this distribution, and train the decoder to create a _similar_ image as the input.

![VAE](/vae.png)

The hypothesis here is that, we want the compressed representation from an autoencoder to truly represent the latent variables of the distribution. That is that the latent space \\( z \\) is a smaller set of
factors of variation in our data. eg. for MNIST we have a notion of scale, rotation, number etc. We view these latent variables as being from some distribution themselves, so if we were to sample from the distributions governing the latents they would map to some meaningful data back in the input space.


## Illustrative Example
Imagine we train a VAE on handwritten digits (MNIST). Suppose an image of "3" is encoded into:

\\[
  μ=[0.5,−0.3],  σ=[0.2,0.7]
\\]

This means that the latent representation for "3" isn't just a single point 
\\( [0.5,−0.3] \\) , but a region in latent space where similar "3"-like digits exist.

Now, when generating new digits:

We sample different 
z values from the distribution 

\\( 𝑁(𝜇 , 𝜎^2) \\)

This creates new variations of "3", such as:

A slightly tilted "3" (z=[0.6,−0.2])
A bold "3" (z=[0.4,−0.5])

If 𝜎 σ were too small, the latent space would collapse into fixed points, making the model deterministic (like a regular Autoencoder).

If 𝜎 σ were too large, the samples would be too random, leading to meaningless outputs.


## Psuedo Code

``` python

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super(VAE, self).__init__()

        # Encoder: Compress input into mu and log_sigma
        self.fc1 = nn.Linear(input_dim, 400)  # First layer
        self.fc_mu = nn.Linear(400, latent_dim)  # Mean output
        self.fc_log_sigma = nn.Linear(400, latent_dim)  # Log variance output

        # Decoder: Convert latent space back to image
        self.fc2 = nn.Linear(latent_dim, 400)  
        self.fc3 = nn.Linear(400, input_dim)  # Reconstruct image


    ## Encoder output mu, and sigma and not a single point
    def encode(self, x):
        """ Encode input to mu and log_sigma """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        return mu, log_sigma



    ## This function is used to sample from the output dimension, and also enable gradients
    def reparameterize(self, mu, log_sigma):
        """ Apply reparameterization trick """
        sigma = torch.exp(0.5 * log_sigma)  # Convert log variance to standard deviation
        epsilon = torch.randn_like(sigma)  # Sample from normal distribution
        return mu + sigma * epsilon  # Reparametrization trick

    ## The decoder outputs recostructed input
    def decode(self, z):
        """ Decode from latent space back to original dimension """
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))  # Sigmoid activation for pixel values


    ## Forward pass, encodes, samples, and decodes
    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)  # Sample latent vector
        x_reconstructed = self.decode(z)  # Generate output
        return x_reconstructed, mu, log_sigma

```





# GAN