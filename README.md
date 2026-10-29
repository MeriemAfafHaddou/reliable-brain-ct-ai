# Investigating Model Reliability and Behavior in Brain CT Classification

This project investigates machine learning model behavior beyond classification performance, using intracranial hemorrhage (ICH) detection in brain CT images as a case study.

The experiments explore classification performance, probability calibration, uncertainty estimation, explainability, segmentation-guided learning, and latent-space exploration. The broader motivation is to understand what model predictions reveal about model behavior and where their limitations remain.

## Project Overview

The task is to classify brain CT images into two categories:
- **ICH:** Intracranial hemorrhage
- **Non-ICH:** No intracranial hemorrhage

The project uses ResNet-18 transfer learning and patient-level data splitting.

### Main Components

- Classification and cross-validation
- Sensitivity improvement experiments
- Probability calibration using the Brier score and Expected Calibration Error (ECE)
- Uncertainty estimation using Monte Carlo Dropout
- Grad-CAM visual explanations
- Multichannel CT classification with segmentation information
- Latent-space exploration using a convolutional autoencoder
- Preliminary exploration of diffusion-based image generation

## Notebooks

### `patients_splitting.ipynb`
Constructs and validates patient-level development and held-out test sets.

### `brain_ct_reliability.ipynb`
Covers classification, evaluation, calibration, uncertainty estimation, Grad-CAM, and autoencoder-based latent-space exploration.

### `sensitivity_improvement_experiments.ipynb`
Investigates training strategies to improve ICH sensitivity, including intensity augmentation, oversampling, class-weighted loss, and weight-decay selection.

### `brain_ct_multichannel_segmentation.ipynb`
Explores multichannel CT classification and the use of segmentation information.

### `diffusion_ct_generation_exploration.ipynb`
Provides an initial exploration of diffusion-based generative modeling for CT imaging.

## Dataset

The project uses the [Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation dataset](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images/data).

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks using Jupyter. Ensure the dataset paths are configured for your environment.

## Challenges

- Improving ICH sensitivity while maintaining specificity.
- Evaluating probability calibration and uncertainty estimates.
- Investigating the regions highlighted by visual explanations.
- Exploring latent representations while preserving relevant image features.

## Presentation

The project presentation is available in the [`presentation/`](presentation/) directory.

## Future Work

Potential directions include evaluating generalization on independent datasets, investigating model behavior under distribution shift, quantitatively evaluating explanations, and exploring more effective approaches to counterfactual generation.
