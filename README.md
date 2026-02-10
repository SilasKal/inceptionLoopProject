# Inception Loop Project

This project implements and evaluates a deep learning pipeline inspired by the
*Inception Loop* framework to model neural responses to visual stimuli and to
generate images that maximally excite neurons.

The work combines computational neuroscience and deep learning, focusing on
predictive modeling, generalization analysis, and stimulus optimization.

## Motivation
Understanding how neurons respond to complex visual stimuli is a central problem
in computational neuroscience. Recent work suggests that deep neural networks can
be used not only to predict neural responses, but also to synthesize stimuli that
optimally activate specific neurons or neural populations.

This project investigates the feasibility and limitations of such an approach on
both biological and simulated neural data.

## Approach
The pipeline follows three main steps:
1. Train neural network models to predict neural responses from natural images
2. Evaluate generalization using cross-validation and multiple model architectures
3. Optimize an initially random image via gradient ascent to maximize predicted
   neural activation ("most exciting input")

## Methods
- Data:  
  - Calcium imaging data from ferret visual cortex (V1)  
  - Simulated neural responses based on a data-driven V1 model
- Models:
  - Convolutional Neural Networks (CNNs)
  - Fully connected neural networks (shallow and deep)
  - Transformer-based models (exploratory)
- Techniques:
  - PCA-based dimensionality reduction
  - Pixel selection strategies
  - Regularization, dropout, batch normalization
  - k-fold cross-validation
  - Gradient-based stimulus optimization

## References
Inspired by:
Walker et al., *Inception loops discover what excites neurons most*, Nature Neuroscience (2019)
