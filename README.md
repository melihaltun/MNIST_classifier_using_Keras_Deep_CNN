# MNIST_classifier_using_Keras_Deep_CNN
A deep CNN implementation for solving MNIST digit recognizer dataset, which achieves 99% accuracy.

The code needs tensorflow, keras, numpy, pandas and matplot lib installed in the Python environment.

Download MNIST data from: https://www.kaggle.com/c/digit-recognizer

Copy train.csv file to "./digits_data/" folder under the project folder

During the first run the code will do the train, test, validation data split. They will be reused in the subsequent runs.
Split ratio is set to 90% training, 6% validation and 4% testing. It can be modified as needed.
If a new train, validation & test set is needed, delete the "/train_test_val/" folder under "./digits_data/"

GPU parallelization is turned off, but it can be turned on by uncommenting the relevant line. 

![image](https://user-images.githubusercontent.com/40482921/231029172-6d924ac9-3342-463e-a778-15310f1c0a9d.png)

