🌱 Plant Disease Detection using Convolutional Neural Networks (CNN)

This project builds a deep learning–based system to detect plant diseases from leaf images. Using the PlantVillage dataset, the model classifies input images into healthy or diseased classes using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.

📌 Project Overview

Plant diseases significantly affect crop yield and food supply worldwide. Early detection allows farmers to take timely action and prevent crop loss.

This project:

Loads and preprocesses labeled leaf images.

Builds a CNN capable of learning disease patterns.

Trains the model on processed images using data augmentation.

Evaluates accuracy and loss.

Saves label encoders/models for downstream use.

Provides utilities to convert new images for inference.

🗂️ Dataset

The project uses the PlantVillage dataset, which contains thousands of labeled plant leaf images.

Steps in the notebook:

Download the dataset from Google Drive.

Unzip and explore folder structure.

Extract image paths and generate labels.

Encode labels using LabelBinarizer.

🧪 Data Preprocessing

Key preprocessing steps:

Resize images to a fixed size: 256 × 256

Convert images to arrays using img_to_array

Normalize pixel values

Split into training and testing sets

Save label transformer (.pkl) for later inference

The function:

convert_image_to_array(image_dir)


handles reading and resizing each image safely.

🧠 Model Architecture

The project uses a Sequential CNN, including:

Conv2D layers for feature extraction

BatchNormalization for training stability

MaxPooling2D to reduce spatial dimensions

Dropout to reduce overfitting

Dense layers for classification

The final layer outputs the predicted class (disease type).

🔧 Training

Training is performed using:

Adam optimizer

Binary crossentropy loss

ImageDataGenerator for augmentation

Configurable hyperparameters like:

Learning rate (LR)

Batch size

Number of epochs

Training code:

history = model.fit_generator(
    augment.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

📈 Model Performance

The notebook plots:

Training vs Validation Accuracy

Training vs Validation Loss

These help visualize overfitting and model convergence.

💾 Saving Artifacts

The following files are saved for inference:

plant_disease_label_transform.pkl – Label encoder

Trained model (if saved manually by user)

🔍 Predicting on New Images

The project includes helper functions to:

Load an image

Convert it to the correct shape

Run prediction through the trained model

Map output back to disease label

📦 Requirements

Ensure the following are installed:

numpy
opencv-python
matplotlib
keras
tensorflow
sklearn
pickle


Some cells also install specific versions (e.g., numpy==1.18.0).

▶️ How to Run

Download or clone this project repository.

Open the notebook in Jupyter Notebook or Google Colab.

Run all cells in order.

Use the trained model to classify new plant leaf images.
