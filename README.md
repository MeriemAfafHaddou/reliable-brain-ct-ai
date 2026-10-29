# Investigating Model Reliability in Brain CT Classification

This project investigates machine learning model behavior beyond classification performance, using intracranial hemorrhage (ICH) detection in brain CT images as a case study.

The work explores classification performance, probability calibration, uncertainty estimation, explainability, and segmentation-guided learning. A recurring question is whether model predictions and confidence estimates reflect reliable behavior, particularly when the model makes mistakes.

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
- Multi-window CT classification using brain and bone windows
- Segmentation-guided learning for ICH classification and localization
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
Explores brain and bone CT windows alongside hemorrhage segmentation masks as localization supervision. The experiments investigate whether additional image information and explicit localization supervision can help detect hemorrhages missed by the classifier.

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

## Challenges and Findings

- Improving ICH sensitivity while maintaining specificity.
- Investigating incorrect predictions with high confidence and low Monte Carlo Dropout variability.
- Examining model attention using Grad-CAM and comparing it with available hemorrhage segmentation masks.
- Exploring whether multiple CT windows and localization supervision can help address missed hemorrhages.

The experiments highlight that classification performance, confidence estimates, and visual explanations provide different perspectives on model behavior. None alone fully explains why a model fails.

## Presentation

The project presentation is available in the [`presentation/`](presentation/) directory.

## Future Work

Potential directions include evaluating generalization on independent datasets, investigating model behavior under distribution shift, comparing single-window and multi-window models, and further evaluating uncertainty estimates and explanations. Additional work could explore segmentation-based evaluation and counterfactual generation.
