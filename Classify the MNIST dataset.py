from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import sys

# This file yields an overall accuracy of .913, 8's is .868 with 849 appropriately classified
#region Grabbing and Splitting the Data

# grab the MNIST data and store to a temporary location
# catch the error for frequent HTTP Error 500's
try:
    # use this for temporary storage, but will have to re-download each run.
    # test_data_home = tempfile.mkdtemp()
    # mnist = fetch_mldata('MNIST original', data_home=test_data_home)

    # this caches the data so it only downloads the first run
    mnist = fetch_mldata('MNIST original')
except Exception as ex:
    print(str(ex) + "\nI don't know... this happens sometimes. Run Again.")
    sys.exit(1)

# Print to show there are 70000 images (28 by 28 images for a dimensionality of 784)
print("Image Data Shape", mnist.data.shape)

# split the data into a training and test set
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target,
                                                            test_size=1/7.0, random_state=0)

# showing the images and the labels
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

#endregion

#region Model Training and Prediction

# make an instance of the model
# we will use L-BFGS optimization
logisticRegr = LogisticRegression(solver='lbfgs')

# train the model on the data (model learns the relationship btwn the image and true number)
logisticRegr.fit(train_img, train_lbl)

# using this trained model, predict the labels of the test set
# returns a numpy array

# predict the entire test set
fullPrediction = logisticRegr.predict(test_img)

#endregion

#region Measuring Model Performance

# accuracy as fraction of correct predictions
# calculated by correct predictions/total number of predictions
# this can be done using the score method
score = logisticRegr.score(test_img, test_lbl)

# we can also use a confusion matrix to visualize the performance of a classification model
cm = metrics.confusion_matrix(test_lbl, fullPrediction)
# normalize the confusion matrix to show percentages instead of counts
cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plot this matrix with counts
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Overall Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
plt.show()

# plot this matrix with percentages
plt.figure(figsize=(8, 8))
sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'BuGn_r');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Overall Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
plt.show()

#endregion

#region  Display Misclassified Eights

# as part of the assignment, we want to improve our classification accuracy for 8's
# to get idea's for useful features, let's look at the 8's that were misclassified

index = 0
misclassifiedIndexes = []
for label, predict in zip(test_lbl, fullPrediction):
    if label == 8 and label != predict:
        misclassifiedIndexes.append(index)
    index += 1

print("There were " + str(len(misclassifiedIndexes)) + " misclassified 8's")
print(mnist.data[misclassifiedIndexes[0]])

plt.figure(figsize=(20, 4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(test_img[badIndex][0:784], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(fullPrediction[badIndex],
                                                 test_lbl[badIndex]), fontsize=15)
plt.show()

#endregion
