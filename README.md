# Mini-Project-3-Classification-Of-Eurosat-Images-Using-CNN


The aim of this project is classification of satellite images belonging to 10 classes.

# The Data

The dataset contains 27,000 images, which I divided between 'train', 'validation', and 'test' parts,
which contain 18,900, 4,050, and 4,050 images respectivly.
The images have shape of (64x64x3)
Here is the link to the dataset - https://www.tensorflow.org/datasets/catalog/eurosat

And here are examples of images

![image examples](https://github.com/SargisArzumanyan/Mini-Project-3-Classification-Of-Eurosat-Images-Using-CNN/assets/82839525/239fa8ba-95c8-450f-9fac-9a7fb0741175)

# How To Use The Model

Move to the 'feature' branch of this repository, download the notebook and eurosat_model.keras files,
run the notebook using jupyter or google colab. If you don't want to train the model again, upload 'eurosat_model.keras' file and uncomment 'new_model = tf.keras.models.load_model('eurosat_model.keras')' line, and test the model.

# Modules used

I used 'tensorflow' and 'tensorflow.keras' libraries for creating this neural network model, 'tensorflow_datasets' module to upload and split the data, 'matplotlib' for visualization, 'numpy' and 'pandas' are also imported, but are not used.

# Data Preparation

I shuffled the training set, rescaled the values of pixels of all 3 sets to the range of [0:1] , and divided them into batches of size 32.

# The Model

This is a sequential model, which containes 4 'Conv2D' convolutional layers containing 32, 64, 128, 256 neurons respectivly and each has kernel size of 3 x 3, 'relu' activation, and padding is set to 'same'. 
The model contains 3 'MaxPooling2D' layers, with pool size 2x2, and 1 'GlobalAvgPool2D' layer.
There is 1 dropout layer with dropout rate of 0.375. 
And finally it has 1 'Dense' output layer with 10 neurons (equal to number of classes), and 'softmax' activation function.

Here is a sumarry of the model.
![model summary](https://github.com/SargisArzumanyan/Mini-Project-3-Classification-Of-Eurosat-Images-Using-CNN/assets/82839525/f6b42982-e45d-44d9-8d1d-44db04c4b2c7)


# The Training

I used 'AdamW' optimizer with learning rate of 0.001, and the loss is 'sparse categorical crossentropy'.
I used early stopping for regularization with 25 patience, which monitored the validation accuracy.
I set the 'epochs' equal to 250, but it stopped earlier because of no improvement after 141 epochs.

# The Result

The model has an accuracy of 0.9568 on a test set.
![image](https://github.com/SargisArzumanyan/Mini-Project-3-Classification-Of-Eurosat-Images-Using-CNN/assets/82839525/96d6198c-9acd-4661-a570-9b468b0885f1)

You can see the training process here

![training history](https://github.com/SargisArzumanyan/Mini-Project-3-Classification-Of-Eurosat-Images-Using-CNN/assets/82839525/f1c7f574-6c2e-4dbb-8501-f540c50ec681)

# What else I have tried.

I tried 5 different values for dropout rate, tried to add more 'dense' and 'dropout' layers and to use 'flatten' instead of 'global average pooling', also tweaked the learning rate.

