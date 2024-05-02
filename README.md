# confounder-free-cnn

## CF-net-2-labels Notebook Overview

### Description

The `CF-net-2-labels.ipynb` notebook presents an implementation of a machine learning model designed to predict Age-related Macular Degeneration (AMD) from OCTDL dataset images. The model leverages a Convolutional Neural Network (CNN) architecture and incorporates a regressor to mitigate the influence of age as a confounder, following the approach outlined in [BR-Net by Qingyu Zhao et al.](https://github.com/QingyuZhao/BR-Net).

### Objective

The primary goal of this notebook is to develop a predictive model that can effectively distinguish between patients with and without AMD while addressing the bias introduced by age variations among patients. By making the images "confounder-free," the model aims to improve the accuracy and reliability of AMD diagnosis based on OCTDL dataset images.

### Methodology

1. **Data Preprocessing**: The notebook begins with preprocessing steps where OCTDL images are loaded and preprocessed to suit the model's input requirements.
2. **Confounder Handling**: It implements a regression model to predict and subsequently remove the influence of the confounding variable (age) from the image features. This step ensures that the subsequent predictions focus purely on disease-relevant features rather than age-related variations.
3. **CNN Model**: Post confounder adjustment, a CNN model is employed to classify images into two categories: AMD and no AMD. The model architecture is inspired by the VGG16 model, adapted to handle the specific characteristics of OCTDL images.
4. **Training and Evaluation**: The model is trained on the adjusted features, and its performance is evaluated using accuracy metrics and a confusion matrix to provide insights into its predictive capabilities.

### Implementation

- The implementation utilizes TensorFlow and Keras for building and training the neural network models.
- The `plot_confusion_matrix` function is used to visually assess the model's performance and understand its effectiveness in distinguishing between the two classes.

### Results

The notebook provides a detailed analysis of the model's accuracy and the effectiveness of the confounder adjustment technique. Results are visualized through various plots, including the confusion matrix, to offer intuitive insights into the model's classification prowess.

### Usage

To run this notebook:

1. Ensure that all required Python packages mentioned in the `requirements.txt` file are installed.
2. Open the notebook in a Jupyter environment and execute the cells sequentially.

### Future Work

Further improvements might involve optimizing the CNN architecture, experimenting with different methods of confounder adjustment, or applying the model to a broader set of imaging data to validate its robustness and generalizability.

Certainly! Here's a polished version of the description for your README file, formatted to clearly articulate the purpose, content, and methodology employed in the `OCTDL CNN_2_classes.ipynb` notebook:

---

## OCTDL CNN_2_classes.ipynb Overview

### Introduction

The `OCTDL CNN_2_classes.ipynb` notebook is a comprehensive guide to applying Convolutional Neural Networks (CNNs) for distinguishing between two specific conditions: Age-related Macular Degeneration (AMD) and Non-AMD, utilizing the OCTDL dataset. This notebook aligns with the methodologies outlined in the [OCTDL repository](https://github.com/mikhailkulyabin/octdl).

### Objective

The primary goal of this notebook is to leverage deep learning techniques, particularly CNNs, to enhance the detection of AMD in medical diagnostics. By accurately classifying OCTDL images into AMD and Non-AMD categories, this work contributes to advancing diagnostic processes and supporting early detection efforts.

### Methodology

1. **Data Preprocessing**: Initial steps involve loading and preprocessing the OCTDL images to fit the input requirements of the CNN model. This includes normalization, resizing, and augmenting the data to ensure robustness and mitigate overfitting.

2. **Model Configuration**: The notebook outlines the setup of the CNN architecture tailored for the task. The configuration details such as layer types, activation functions, and other hyperparameters are meticulously chosen to optimize performance.

3. **Training the Model**: Detailed steps guide the user through the process of training the model with the preprocessed dataset. Techniques to optimize the training process, such as adjusting learning rates and using callbacks for model improvements, are discussed.

4. **Performance Evaluation**: After training, the model's effectiveness is evaluated using accuracy metrics and a confusion matrix. These tools help visualize and understand the model’s capability to differentiate between AMD and Non-AMD conditions accurately.

### Results

The results section provides insights into the accuracy of the model, highlighting its success in identifying and categorizing the conditions based on the OCTDL dataset. Graphical representations of training and validation loss and accuracy are included to demonstrate the model's learning curve.

Certainly! Here’s a revised description for your README file, making sure to correct the count of labels (from two to three) and adjusting the content accordingly for the `OCTDL CNN_3_classes.ipynb` notebook:

---

## OCTDL CNN_3_classes.ipynb Overview

### Introduction

The `OCTDL CNN_3_classes.ipynb` notebook demonstrates the application of Convolutional Neural Networks (CNNs) for classifying optical coherence tomography images from the OCTDL dataset into three distinct classes: Age-related Macular Degeneration (AMD), Epiretinal Membrane (ERM), and Normal. This approach follows the methodologies provided in the [OCTDL repository](https://github.com/mikhailkulyabin/octdl).

### Objective

This notebook aims to enhance the diagnostic capabilities of medical imaging systems by employing deep learning techniques to accurately identify and differentiate among AMD, ERM, and Normal conditions. The goal is to contribute to the improvement of diagnostic accuracy and the efficiency of medical treatments through advanced image analysis.

## OCTDL Data

OCTDL data can be download from **https://data.mendeley.com/datasets/sncdhf53xc/4**

## Environment

**The environment.yml is used to initialize a python 3.6 environment to run the cfnet notebook. For the other files, apython 3.11 can be used.**
