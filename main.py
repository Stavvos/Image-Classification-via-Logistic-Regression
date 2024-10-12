import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, cross_val_score 
from tensorflow.keras.datasets import mnist
from skimage import feature
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

def loadData():
    print("\n\nLoading, splitting, and normalising data")
    #import the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("\n\nThe shape of the training dataset before split: ", x_train.shape, " and their labels: ", y_train.shape)
    print("\n\nThe shape of the testing dataset before split: ", x_test.shape, " and their labels: ", y_test.shape)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    #split up the data into a 80/20 split: 80 for training, 20 for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    #normalize the data
    #normalise the data. you could do this by dividing the values by 255.
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)

    #reshape the data 
    numberOfImagesTrain = len(x_train)
    numberOfImagesTest = len(x_test)
    pixels = 28
    channels = 1
    x_train = x_train.reshape((numberOfImagesTrain, pixels, pixels, channels))  
    x_test = x_test.reshape((numberOfImagesTest, pixels, pixels, channels))  


    print("\n\nThe shape of the training data after split: ", x_train.shape, " and the labels: ", y_train.shape)
    print("\n\nThe shape of the testing data after split", x_test.shape, "and the labels: ", y_test.shape, "\n\n")


    return x_train, y_train, x_test, y_test

def augmentData(x_train, y_train, x_test, y_test, dataAugmented):
    print("\n\nAugmenting data")
    # Create an ImageDataGenerator object with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,       # Rotate the image by up to 10 degrees
        width_shift_range=0.1,   # Shift the image horizontally by 10%
        height_shift_range=0.1,  # Shift the image vertically by 10%
        zoom_range=0.1,          # Zoom in by up to 10%
        shear_range=0.1,         # Shear the image by 10%
        horizontal_flip=False,   # No horizontal flip for digit images (since MNIST digits are directional)
        fill_mode='nearest'      # Filling in pixels with nearest pixel value
    )

    # Fit the generator to the training data 
    datagen.fit(x_train)
    datagen.fit(x_test)

    #display the first image from x_train and its label
    plt.imshow(x_train[0])
    plt.title(f"The label for the first training image after augmentation is: {y_train[0]}")
    plt.show()

    #display the first image from x_test and its label
    plt.imshow(x_test[0])
    plt.title(f"The label for the first testing image after augmentation is: {y_test[0]}")
    plt.show()

    print("\n\nThe shape of the training data after augmentation: ", x_train.shape, " and the labels: ", y_train.shape)
    print("\n\nThe shape of the testing data after augmentation: ", x_test.shape, "and the labels: ", y_test.shape, "\n\n")
    
    dataAugmented = True

    return x_train, x_test, dataAugmented

def extractHogFeatures(x_train, y_train, x_test, y_test, HOG):
    print("\nExtracting the HOG features")

    hogFeaturesTrain = []
    hogImagesTrain = []
    hogFeaturesTest = []
    hogImagesTest = []

    #get HOG features from training set
    for i in range(len(x_train)):
        image = x_train[i].squeeze()

        hogFeature, hogImage = feature.hog(image,
                                            orientations = 9,
                                            pixels_per_cell = (8,8),
                                            cells_per_block = (2,2),
                                            visualize = True)
        hogFeaturesTrain.append(hogFeature)
        hogImagesTrain.append(hogImage)

    #get HOG features from testing set
    for i in range(len(x_test)):
        image = x_test[i].squeeze()

        hogFeature, hogImage = feature.hog(image,
                                            orientations = 9,
                                            pixels_per_cell = (8,8),
                                            cells_per_block = (2,2),
                                            visualize = True)
        hogFeaturesTest.append(hogFeature)
        hogImagesTest.append(hogImage)
    
    #plot the first image from training set, and then that same image as HOG features
    fig, ax = plt.subplots(1,2,figsize = (12, 6))
    ax[0].imshow(x_train[0], cmap='gray')
    ax[0].set_title(f"Original Image. Label = {y_train[0]}")
    ax[0].axis('off')

    ax[1].imshow(hogImagesTrain[0], cmap='gray')
    ax[1].set_title(f"HOG Features. Label = {y_train[0]}")
    ax[1].axis('off')

    plt.show()

    #plot the first image from testing set, and then that same image as HOG features
    fig, ax = plt.subplots(1,2,figsize = (12, 6))
    ax[0].imshow(x_test[0], cmap='gray')
    ax[0].set_title(f"Original Image. Label = {y_test[0]}")
    ax[0].axis('off')

    ax[1].imshow(hogImagesTest[0], cmap='gray')
    ax[1].set_title(f"HOG Feature. Label = {y_test[0]}")
    ax[1].axis('off')

    plt.show()

    HOG = True

    return hogFeaturesTrain, hogFeaturesTest, HOG

def trainModel(x_train, y_train, x_test, y_test, dataAugmented, HOG):
    print("\n\nTraining the logistic regression model\n\n")
    
    if(dataAugmented == True and HOG == False):
        #reshape the data
        x_train = x_train.reshape(56000, -1)  # Flatten images to 1D
        x_test = x_test.reshape(14000, -1)  # Flatten test images

    #initialize and train the model
    model = LogisticRegression(solver = 'lbfgs')
    model.fit(x_train, y_train)

    #make some predictions with the newly trained model
    predictions = model.predict(x_test)

    #perform cross validation on the model
    print("\n\nPerforming cross validation on the logisitic regression model\n\n")
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    crossValScores = cross_val_score(model, x, y, cv = 5)

    #print the results of the cross validation
    print("\nThe cross validation scores are: ", crossValScores)
    print("The mean accuracy of the cross validation scores is:", crossValScores.mean(), " with a standard deviation of: ", crossValScores.std())


    #initialise the confusion matrix
    confusionMatrix = confusion_matrix(y_test, predictions)

    #make a classification report and print it to the console
    report = classification_report(y_test, predictions, target_names = [str(i) for i in range(10)])
    print("\nThe classification report for the logistic regression model is:\n", report)

    #make a simple performance report
    print("\nA performance report for the logistic regression model classifying hand written digits:")
    print("Accuracy:",accuracy_score(y_test, predictions))
    print("Precision:",precision_score(y_test, predictions, average = 'weighted'))
    print("Recall:",recall_score(y_test, predictions, average = 'weighted'))
    print("F1-score:",f1_score(y_test, predictions, average = 'weighted'))
    print("\nconfusion matrix:\n", confusionMatrix)





def main():
    dataAugmented = False
    HOG = False
    print("\n\nmain running")
    x_train, y_train, x_test, y_test = loadData()
    x_train, x_test, dataAugmented = augmentData(x_train, y_train, x_test, y_test, dataAugmented)
    x_train, x_test, HOG = extractHogFeatures(x_train, y_train, x_test, y_test, HOG)
    trainModel(x_train, y_train, x_test, y_test, dataAugmented, HOG)

main()
