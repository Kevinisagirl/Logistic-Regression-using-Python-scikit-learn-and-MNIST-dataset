from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import sys

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

np.set_printoptions(threshold=np.nan)

print(type(mnist))
print(len(mnist.data))
print(len(mnist.data[0]))
newmnist = np.zeros(shape=(70000, len(mnist.data[0])+2))
print(len(newmnist[0]))
for a in range(0, len(mnist.data)):
    newmnist[a] = np.append(mnist.data[a], [int(np.mean(mnist.data[a][0:28*14])),
                            int(np.mean(mnist.data[a][28*14:]))])

mnist.data = newmnist
print(mnist.data[0])
