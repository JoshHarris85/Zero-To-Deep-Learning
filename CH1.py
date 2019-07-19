import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import rcParams
from IPython.display import set_matplotlib_formats

a = np.array([1, 3, 2, 4])

a
type(a)

b = np.array([[8, 5, 6, 1],
              [4, 3, 0, 7],
              [1, 3, 2, 9]])

c = np.array([[[1, 2, 3],
               [4, 3, 6]],
              [[8, 5, 1],
               [5, 2, 7]],
              [[0, 4, 5],
               [8, 9, 1]],
              [[1, 2, 6],
               [3, 7, 4]]])

# 1d array = vector
# 2d array = matrix
# 3d array = tensor of order 3
# List of arrays of equal shape

a.shape
b.shape
c.shape

# Select first column of b with :
b[:, 0]
# Returns array([8, 4, 1])

# we can select the upper left 2x2 sub-matrix in b as:
b[:1, :1]
# array([[8]])

# Select the second and third elements of a:
assert ((a[1:3] == np.array([3, 2])).all())

# Select the elements from 1 to the end in a:
assert ((a[1:] == np.array([3, 2, 4])).any())

# Select all the elements from the beginning excluding the last one:
assert ((a[:-1] == np.array([1, 3, 2])).all())

# swaps the rows with the columns
b.transpose()

# Matplotlib example
rcParams['font.size'] = 14
rcParams['lines.linewidth'] = 2
rcParams['figure.figsize'] = (7.5, 5)
rcParams['axes.titlepad'] = 14
rcParams['savefig.pad_inches'] = 0.12
set_matplotlib_formats('png', 'pdf')

plt.plot(a);
plt.plot(a, 'o');
plt.plot(b, 'o-');

plt.figure(figsize = (9, 6))

plt.plot(b[0], color='green', linestyle='dashed',
         marker='o', markerfacecolor='blue',
         markersize=12 )
plt.plot(b[1], 'D-.', markersize=12 )

plt.xlabel('Remember to label the X axis', fontsize=12)
plt.ylabel('And the Y axis too', fontsize=12)

t = r'Big Title/greek/math: $\alpha \sum \dot x \^y$'
plt.title(t, fontsize=16)

plt.axvline(1.5, color='orange', linewidth=4)
plt.annotate(xy=(1.5, 5.5), xytext=(1.6, 7),
             s="Very important point",
             arrowprops={"arrowstyle": '-|>'},
             fontsize=12)
plt.text(0, 0.5, "Some Unimportant Text", fontsize=12)

plt.legend(['Series 1', 'Series 2'], loc=2);

# Scikit-learn is a beautiful library for many Machine Learning algorithms in Python. We will use it here to generate some data
from sklearn.datasets import make_circles

# The make_circle function will generate two "rings" of data points, each with two coordinates. It will also create an array of labels, either 0 or 1.
X, y = make_circles(n_samples=1000,
                    noise=0.1,
                    factor=0.2,
                    random_state=0)

# X indicates the input variable, and it is usually an array of dimension >= 2 with the outer index running over
# the various data points in the set.
# y indicates the output variable, and it can be an array of dimension >= 1. In this case, our data will belong to
# either one circle or the other, and therefore our output variable will be binary: either 0 or 1. In particular,
# the data points belonging to the inner circle will have a label of 1.
X.shape
y.shape

plt.figure(figsize=(5, 5))
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])
plt.title("Blue circles and Red crosses");

# Machine learning with Keras
# Seperate blue dots from red crosses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Create shell model tell it we will build our model adding one component at a time (sequentially)
model = Sequential()

model.add(Dense(4, input_shape=(2,), activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=SGD(lr=0.5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=20);

hticks = np.linspace(-1.5, 1.5, 101)
vticks = np.linspace(-1.5, 1.5, 101)

aa, bb = np.meshgrid(hticks, vticks)

aa.shape

plt.figure(figsize=(5, 5))
plt.scatter(aa, bb, s=0.3, color='blue')
# highlight one horizontal series of grid points
plt.scatter(aa[50], bb[50], s=5, color='green')
# highlight one vertical series of grid points
plt.scatter(aa[:, 50], bb[:, 50], s=5, color='red');

# The model expects a pair of values for each data point, so we have to re-arrange aa and bb into a single array with two columns.
# The ravel function flattens an N-dimensional array to a 1D array, and the np.c_ class will help us combine aa and bb into a single 2D array.
ab = np.c_[aa.ravel(), bb.ravel()]

ab.shape

c = model.predict(ab)

c.shape
cc = c.reshape(aa.shape)
cc.shape

plt.figure(figsize=(5, 5))
plt.scatter(aa, bb, s=20*cc);

plt.figure(figsize=(5, 5))
plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
plt.title("Blue circles and Red crosses");
