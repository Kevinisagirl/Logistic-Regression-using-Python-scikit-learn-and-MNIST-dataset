from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import sys
from progress.bar import Bar

#region Grabbing the Data

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

#endregion

#region Adding features

# create a new array of the correct sizes to store the new feature values
newmnist = np.zeros(shape=(70000, len(mnist.data[0])+29))

#region average of top and average of bottom <- overall accuracy = 0.9128, '8' up by 0.001
# loop through the mnist data and find the average of the top and bottom halves
'''
for a in range(0, len(mnist.data)):
    newmnist[a] = np.append(mnist.data[a], [int(np.mean(mnist.data[a][0:28*14])),
                            int(np.mean(mnist.data[a][28*14:]))])
'''
#endregion

#region total of top and total of bottom handwriting <- overall accuracy = 0.9097 , '8' classified 837
# loop through the mnist data and find the total of the top and bottom halves
# find top and bottom first by finding the first and last rows of handwriting
'''
for a in range(0, len(mnist.data)):
    first = 10000000
    last = 0
    for pixel in range(1, len(mnist.data[a])):
        if mnist.data[a][pixel] != 0:
            if pixel < first:
                first = pixel
            if pixel > last:
                last = pixel
    middle = int((last-first)/2)

    newmnist[a] = np.append(mnist.data[a], [sum(mnist.data[a][0:middle]),
                            sum(mnist.data[a][middle:])])
'''
#endregion

#region average of top and total of bottom handwriting <- doesn't work, issue with numpy.mean
# loop through the mnist data and find the average of the top and bottom halves
# find top and bottom first by finding the first and last rows of handwriting
'''
for a in range(0, len(mnist.data)):
    first = 0
    last = 0
    for pixel in range(1, len(mnist.data[a])):
        if mnist.data[a][pixel] != 0:
            if first == 0:
                first = pixel
            if pixel > last:
                last = pixel
    if last < first:
        print(mnist.data[a])
    middle = int((last-first)/2)
    newmnist[a] = np.append(mnist.data[a], [np.mean(mnist.data[a][first:middle]),
                            np.mean(mnist.data[a][middle:last])])
'''
#endregion

#region count how many lines go through each row <- overall accuracy = .913, '8' classified 845
# this will be approximated by how many numbered pixels are next to a zero per row
# for each handwritten image
'''
for a in range(0, len(mnist.data)):
    # empty array to hold number of handwritten lines passing through each row
    rowlinecounts = np.zeros(28)
    # for each row
    for row in range(0, 28):
        # for each column/pixel in each row
        for pixel in range(0, 28):
            # the index that pixel is located at in the original array
            pixelindex = row*28+pixel
            # if the pixel contains some handwriting
            if mnist.data[a][pixelindex] != 0:
                # and this handwriting borders an edge, update the counter for that row
                if mnist.data[a][pixelindex-1] == 0:
                    rowlinecounts[row] += 1
                if mnist.data[a][pixelindex+1] == 0:
                    rowlinecounts[row] += 1
    # add this array of number of handwritten lines passing through the rows to the end of the dataset
    newmnist[a] = np.append(mnist.data[a], rowlinecounts)
'''
#endregion

#region count how many lines go through each row method 2 <- overall accuracy = 913, '8' classified 846
# this will be approximated by counting increases and decreases in pixel intensity
'''
for a in range(0, len(mnist.data)):  # for each handwritten image
    # empty array to hold number of handwritten lines passing through each row
    rowlinecounts = np.zeros(28)
    for row in range(0, 28):  # for each row
        increasing = True
        for pixel in range(0, 27):  # for each column/pixel in each row
            pixelindex = row*28+pixel  # the index that pixel is located at in the original array
            currentshade = mnist.data[a][pixelindex]
            # if the next pixel is darker and the previous was lighter or equal
            if (mnist.data[a][pixelindex+1] < currentshade) and increasing:
                rowlinecounts[row] += 1
                increasing = False
            if mnist.data[a][pixelindex+1] > currentshade:
                increasing = True
    # add this array of number of handwritten lines passing through the rows to the end of the dataset
    newmnist[a] = np.append(mnist.data[a], rowlinecounts)
'''
#endregion

#region the width of each row normalized by widest width <- overall accuracy = 0.9134, '8s' classified = 849
# this will be approximated by counting increases and decreases in pixel intensity

# to watch percent completion while running
bar = Bar('Processing', max=784)

for a in range(0, len(mnist.data)):  # for each handwritten image
    #print(str(round(a/70000*100)) + '%') if a % 1250 == 0 else 0  # to watch percent completion while running
    bar.next()


    rowwidths = np.zeros(29)  # will hold the width of handwriting in each row
    widestwidth = 0
    widestrow = 0
    rowlinecounts = np.zeros(28)  # will hold number of handwritten lines passing through each row
    switchcounter = 0  # how many times number of lines passing through the row change
    rowhandwritingranges = dict.fromkeys(range(0, 28))

    for row in range(0, 28):  # for each row
        first = 0  # to hold the first pixel with handwriting in each row
        last = 0  # to hold the last pixel with handwriting in each row

        for pixel in range(0, 28):  # for each pixel in the row
            pixelindex = row*28+pixel  # the index that pixel is located at in the original array
            currentshade = mnist.data[a][pixelindex]

            if currentshade != 0:  # if the pixel has handwriting in it
                previousshade = mnist.data[a][pixelindex - 1]
                nextshade = mnist.data[a][pixelindex + 1]
                if (previousshade == 0) or (nextshade == 0):  # if borders an edge, update counter
                    rowlinecounts[row] += 1
                if first == 0:  # if this is the first handwriting in this row, save it's location
                    first = pixelindex
                if pixelindex > last:  # if this is the last handwriting in this row, save it's location
                    last = pixelindex


        rowhandwritingranges[row] = [first, last]
        currentrowwidth = last - first  # how wide was the handwriting in this row
        rowwidths[row] = currentrowwidth  # add the row width to the list
        if currentrowwidth > widestwidth:  # check if we've found a wider row
            widestwidth = currentrowwidth
            widestrow = [row, first, last]

    if widestwidth != 0:
        rowwidths = rowwidths/widestwidth  # normalize the row widths by the widest row

    newmnist[a] = np.append(mnist.data[a], rowwidths)  # add this to the data

    for count in range(1, len(rowlinecounts)):  # for each row
        # if number of lines in row is diff than the row above count this as a switch
        if rowlinecounts[count] != rowlinecounts[count-1]: switchcounter += 1
    # add the switch counter to the data
    newmnist[a][812] = switchcounter

    # # now we need code to process the image
    # # we're going to cut the image in half and left adjust the left and right adjust the right
    # middlehandwriting = widestrow[1] + int(widestwidth/2)  # this is the middle pixel of the widest row
    # for row in range(0,28):
    #     middlepixel = (row - widestrow[0])*28+middlehandwriting  # compute middle pixel of each row
    #     hwstart = rowhandwritingranges[row][0]
    #     hwend = rowhandwritingranges[row][1]
    #     hwlength = hwend - hwstart
    #
    #     if hwstart != 0:  # if our row has some handwriting
    #         leftlenth = middlepixel - hwstart
    #         rightlength = hwend - middlepixel
    #         rowbegin = row * 28
    #         rowend = row * 28 + 27
    #         # if the handwriting in the row falls along the middle axis we need to split that row
    #         if (hwstart < middlepixel) and (hwend > middlepixel):
    #             newmnist[a][rowbegin:rowbegin+leftlenth+1] = mnist.data[a][hwstart:middlepixel +1]
    #             newmnist[a][rowend-rightlength + 1:rowend+1] = mnist.data[a][middlepixel+1:hwend+1]
    #             newmnist[a][rowbegin+leftlenth+1:rowend-rightlength+1] = [0]*(28-leftlenth-rightlength-1)
    #         elif hwstart < middlepixel:  # if all handwriting is to the left of the middle axis
    #             newmnist[a][rowbegin:rowbegin + hwlength + 1] = mnist.data[a][hwstart:hwend + 1]
    #             newmnist[a][rowbegin + hwlength + 1:rowend+1] = [0] * (rowend - rowbegin - hwlength)
    #         else:  # if all the handwriting is to the right of the middle axis
    #             newmnist[a][rowend - hwlength: rowend + 1] = mnist.data[a][hwstart:hwend + 1]
    #             newmnist[a][rowbegin: rowend - hwlength] = [0] * (28 - hwlength - 1)
bar.finish()
#endregion

#region total SWITCHES of lines go through each row <- overall accuracy = 0.9133, '8s' classifed = 849
# this will be approximated by how many numbered pixels are next to a zero per row
# then counting how many times that switches row to row in the image

# for a in range(0, len(mnist.data)):  # for each handwritten image
#     # empty array to hold number of handwritten lines passing through each row
#     rowlinecounts = np.zeros(28)
#     switchcounter = 0
#     for row in range(0, 28):  # for each row
#         for pixel in range(0, 28):  # for each column/pixel in each row
#             pixelindex = row*28+pixel  # index of pixel in original array
#             if mnist.data[a][pixelindex] != 0:  # if the pixel contains some handwriting
#                 if mnist.data[a][pixelindex-1] == 0:  # borders an edge, update counter
#                     rowlinecounts[row] += 1
#                 if mnist.data[a][pixelindex+1] == 0:  # borders an edge, update counter
#                     rowlinecounts[row] += 1
#     for count in range(1, len(rowlinecounts)):  # for each row
#         # if number of lines in row is diff than number of lines passing through the row above
#         if rowlinecounts[count] != rowlinecounts[count-1]:
#             switchcounter += 1  # count this as a switch
#     # add the switch counter to the data
#     newmnist[a][812] = switchcounter

#endregion

#region Center each row <- Overall Accuracy Score = 0.9078, '8s' classified = 896!!! using this one

# for a in range(0, len(mnist.data)):  # for each handwritten image
#     for row in range(0, 28):  # for each row
#         first = 0
#         last = 0
#         for pixel in range(0, 28):  # for each column/pixel in each row
#             pixelindex = row*28+pixel  # index of pixel in original array
#             if mnist.data[a][pixelindex] != 0:
#                 if first == 0:
#                     first = pixelindex
#                 if pixelindex > last:
#                     last = pixelindex
#         rowwidth = last - first
#         if rowwidth != 0:
#             leftsidezeros = np.zeros(shape=(1, (int((28 - rowwidth - 1)/2))))
#             newrowbeginning = np.append(leftsidezeros, mnist.data[a][first:last+1])
#             # this only overwrites some of the data more heavily weighting outskirts... feature not a bug
#             newmnist[a][(row*28):row*28+len(newrowbeginning)] = newrowbeginning
#endregion

print(newmnist[0])

# replace the original mnist with the mnist + new features
mnist.data = newmnist

#endregion

# split the data into a training and test set
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target,
                                                            test_size=1/7.0, random_state=0)

# showing the images and the labels
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image[0:784], (28, 28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

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

# plot this matrix
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

#region  Display Misclassifieds

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
