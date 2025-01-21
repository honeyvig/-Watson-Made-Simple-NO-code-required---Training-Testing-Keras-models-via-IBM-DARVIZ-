# -Watson-Made-Simple-NO-code-required---Training-Testing-Keras-models-via-IBM-DARVIZ-
Watson Made Simple: Training & Testing Keras Models via IBM DARVIZ

IBM DARVIZ (Data Analysis and Visualization) is a tool designed to make machine learning tasks like training and testing models easier, especially for those without a deep technical background. While Watson services such as Watson Machine Learning are typically used for deploying and managing models in a production environment, DARVIZ offers a low-code/no-code interface to interact with machine learning models and datasets.

To integrate Keras models with IBM DARVIZ, we can utilize Watson services for model deployment and testing, enabling the use of a visual interface for managing, training, and testing machine learning models. In this tutorial, we will explain the overall process for training and testing Keras models via IBM DARVIZ.
1. Prerequisites

    IBM Cloud Account: Make sure you have an IBM Cloud account to access Watson services and DARVIZ.
    IBM Watson Studio: Set up Watson Studio to work with IBM DARVIZ. You can use Watson Studio for machine learning, data analysis, and building models.
    Keras Model: We'll use a Keras model for demonstration purposes. If you don't already have a model, you can create one as shown below.
    Dataset: You can use any dataset you prefer (e.g., MNIST, CIFAR-10), or DARVIZ can directly pull data from a CSV or an IBM Cloud storage.

2. Training a Simple Keras Model

First, let's create a simple Keras model. In this case, we’ll use the MNIST dataset for simplicity.

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images to 1D vectors
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Build the Keras model
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the model
model.save('mnist_model.h5')

    This code trains a simple neural network on the MNIST dataset, which is a set of 28x28 pixel images of handwritten digits.
    The model consists of one hidden layer (128 neurons) and an output layer with 10 neurons (for the 10 digits).
    The model is trained using sparse categorical cross-entropy loss and Adam optimizer.

3. Setting Up IBM DARVIZ
Step 1: Create an IBM Cloud Account

    If you don’t already have an IBM Cloud account, go to IBM Cloud and sign up.
    After signing up, create an IBM Watson Studio instance from the IBM Cloud Dashboard.

Step 2: Set Up DARVIZ in IBM Watson Studio

    Go to Watson Studio in the IBM Cloud Dashboard.
    Once you're in the Watson Studio interface, you should see an option to create a new project.
    Choose a project type (e.g., Data Science or Machine Learning) and follow the steps to create the project.
    After creating the project, you can launch DARVIZ (the tool is often integrated directly within the Watson Studio environment).

Step 3: Upload Dataset

You can upload datasets into Watson Studio for DARVIZ to use. DARVIZ can directly access datasets in a variety of formats (CSV, Excel, etc.).

    You can upload the MNIST dataset or use your own dataset for training.
    In Watson Studio, click Add to Project to upload the dataset.

4. Training the Keras Model with DARVIZ (No-Code Interface)

Once you have set up Watson Studio and DARVIZ, follow these steps to train a Keras model through the visual interface:
Step 1: Choose the Dataset

    In DARVIZ, go to Data and select the dataset you wish to use for training (e.g., MNIST or your custom dataset).
    You can perform some data preprocessing in the Data section if needed (like normalizing or splitting the data).

Step 2: Set Up the Model

    Go to the Modeling section within DARVIZ.

    You’ll see an option to Create Model. Choose Keras as your preferred framework.

    DARVIZ provides a drag-and-drop interface where you can build a neural network by selecting layers, activations, and other parameters.

    Choose Dense Layers, ReLU activation, and the softmax output layer for classification.

    You can define the model structure visually, without writing any code. Choose the appropriate input shape and other parameters.

Step 3: Train the Model

    After setting up the model, you can configure the hyperparameters (e.g., learning rate, batch size).
    Click on the Train button in DARVIZ, and the platform will train the Keras model with your dataset.
    DARVIZ will automatically display the training progress, including loss and accuracy graphs, which help you evaluate the model's performance during training.

Step 4: Evaluate the Model

Once the training is complete, you can evaluate the model using the test data. DARVIZ provides metrics such as accuracy, loss, and confusion matrix to help you understand how well the model is performing.
Step 5: Save and Export the Model

    Once you're satisfied with the model, you can save it directly in Watson Studio.
    DARVIZ also allows exporting the trained model for use in deployment, such as through Watson Machine Learning for production use.

5. Testing the Keras Model via IBM DARVIZ

Once the model is trained, you can use DARVIZ's testing interface to evaluate its performance on new data.

    Test the Model with New Data: Upload new test data or use existing test sets.
    Analyze Results: DARVIZ will generate a prediction and show you the results, like the class predicted by the model for each input.

6. Example Workflow for Testing via IBM DARVIZ

Here's how to use the DARVIZ interface for testing:

    Upload test data (e.g., a test image for MNIST or another dataset).
    Click on the Predict button in DARVIZ.
    It will run inference on the test data and output the predicted class label.
    You can review metrics such as precision, recall, and F1-score to assess the model’s performance on unseen data.

7. Deploying the Model

After training and testing your Keras model using DARVIZ, you can deploy it using IBM Watson Machine Learning for integration into applications or APIs. This allows you to create a production-ready environment for your machine learning model.
8. Conclusion

Using IBM DARVIZ, you can train, test, and deploy Keras models without writing code. The interface provides a low-code/no-code approach that allows you to focus on the data and model design rather than managing the underlying code. This is especially useful for business analysts or those who need to prototype machine learning models quickly.

In this tutorial:

    We trained a simple Keras model using the MNIST dataset.
    We used IBM Watson Studio and DARVIZ for training and testing the model via a graphical interface.
