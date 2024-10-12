This project aims to classify images of hand written digits. The program uses a logistic regression model for classification. The MNIST hand written digits
dataset has been used, and it has been split into a 80/20 train test split. Some data preprocessing has been performed which includes data augmentation,
data normalisation, and feature extraction (HOG). The program is set up so that combinations of data augmentation and feature extraction can be used by commenting or
uncommenting the function cals within main(). The model's performance can be determined by the cross validation score, 
classification report, and the confusion matrix. 
