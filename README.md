# Recognizing-hand-written-digits-using-machine-learning

 
# Dataset
Data set used is from UCI Machine learning repository .
Optical Recognition of Handwritten Digits Data Set
The data that we are interested in is made of 8x8 images of digits, let'shave a look at the first 4 images, stored in the `images` attribute of the
 dataset.  If we were working from image files, you could load them using functions in comments.  Note that each image must have the same size. For these images, we know which digit they represent: it is given in the 'target' of the dataset.

# Data Dictionary
Optical Recognition of Handwritten Digits Data Set used preprocessing programs made available by NIST to extract normalized bitmaps of handwritten digits from a preprinted form. From a total of 43 people, 30 contributed to the training set and different 13 to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of 4x4 and the number of on pixels are counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range 0..16. This reduces dimensionality and gives invariance to small distortions. 

# Working
We use Python Scikit-learn package for applying Classification method to classify the digit
1. First Import datasets, classifiers and performance metrics
2. To apply a classifier on this data, we need to flatten the image, to
 3. Turn the data in a (samples, feature) matrix:
 4. Create a classifier: a support vector classifier
5. We learn the digits on the first half of the digits
6. Predict the value of the digit on the second half:
7. now we built confusion matrices and classification report on the basis of our expected and predicted result
8.Finally print our Result

# Requirement
Python Anaconda 
Spyder
