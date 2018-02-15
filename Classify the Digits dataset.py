from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)

# showing the images and the labels
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

# split the data into a training and test set
image_train, image_test, trueNum_train, trueNum_test = train_test_split(digits.data,
                                                                        digits.target,
                                                                        test_size=0.25,
                                                                        random_state=0)
print(trueNum_test[0])

# make an instance of the model
logisticRegr = LogisticRegression()

# train the model on the data (model learns the relationship btwn the image and true number)
logisticRegr.fit(image_train, trueNum_train)

# using this trained model, predict the labels of the test set
# returns a numpy array
# predict for 1 obs
singlePrediction = logisticRegr.predict(image_test[0].reshape(1, -1))
print(singlePrediction)

# predict multiple
multPredictions = logisticRegr.predict(image_test[0:10])
print(multPredictions)

# predict the entire test set
fullPrediction = logisticRegr.predict(image_test)

# measuring model performance
# accuracy as fraction of correct predictions
# calculated by correct predictions/total number of predictions
# this can be done using the score method
score = logisticRegr.score(image_test, trueNum_test)
print(score)

# we can also use a confusion matrix to visualize the performance of a classification model
cm = metrics.confusion_matrix(trueNum_test, fullPrediction)
# use this line to normalize the confusion matrix to show percentages instead of counts
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plot this matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Overall Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
plt.show();