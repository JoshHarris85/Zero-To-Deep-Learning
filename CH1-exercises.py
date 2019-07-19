import numpy as np

# Let's practice a little bit with numpy:

# Exercise 1:
# generate an array of zeros with shape=(10, 10), call it a set every other element of a to 1, both along columns and rows,
# so that you obtain a nice checkerboard pattern of zeros and ones
a = np.zeros((10, 10))

# columns start and end all with step by 2 for x
# rows start and end all with step by 2 for y
a[::2, ::2] = 1

# columns start at 1 and step by 2 for x
# rows start at 1 and step by 2 for y
a[1::2, 1::2] = 1

# generate a second array to be the sequence from 5 included to 15 excluded, call it b
# multiply a times b in such a way that the first row of a is an alternation of zeros and fives, the second row is an alternation
# of zeros and sixes and so on. Call this new array c. To complete this part, you will have to reshape b as a column array
b = np.array(range(5,15))
b =  b.reshape(10, 1)
c = a * b

# calculate the mean and the standard deviation of c along rows and columns
c.mean(axis=0)
c.mean(axis=1)
c.std(axis=0)
c.std(axis=1)

# create a new array of shape=(10, 5) and fill it with the non-zero values of c, call it d
d = c[c>0].reshape(10, 5)

# add random Gaussian noise to d, centered in zero and with a standard deviation of 0.1, call this new array e
noise = np.random.normal(scale=0.1, size=(10, 5))
e = d + noise

# Exercise 2:
# use plt.imshow() to display the array a as an image, does it look like a checkerboard?
plt.imshow(a)

# display c, d and e using the same function, change the colormap to grayscale
plt.imshow(c, cmap='Greys')
plt.imshow(d, cmap='Greys')
plt.imshow(e, cmap='Greys')

# plot e using a line plot, assigning each row to a different data series. This should produce a plot with noisy horizontal lines.
# You will need to transpose the array to obtain this.
plt.plot(e.transpose())

# add a title, axes labels, legend and a couple of annotations
plt.title("Noisy lines")
plt.xlabel("the x axis")
plt.xlabel("the y axis")
plt.annotate(xy=(1, 14), xytext=(0, 12.3),
            s="The light blue line",
            arrowprops={"arrowstyle": '-|>'},
            fontsize=12);

# Exercise 3:
def plot_decision_boundary(model, X, y):
  hticks = np.linspace(X.min()-0.1, X.max()+0.1, 101)
  vticks = np.linspace(X.min()-0.1, X.max()+0.1, 101)
  aa, bb = np.meshgrid(hticks, vticks)
  ab = np.c_[aa.ravel(), bb.ravel()]
  c = model.predict(ab)
  cc = c.reshape(aa.shape)
  plt.figure(figsize=(7, 7))
  plt.contourf(aa, bb, cc, cmap='bwr', alpha=0.2)
  plt.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
  plt.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
  plt.title("Blue circles and Red crosses");

# Exercise 4:
# use the functions make_blobs and make_moons from Scikit-Learn to generate new datasets with two classes
# plot the data to make sure you understand it
# re-train your model on each of these datasets
# display the decision boundary for each of these models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

def train_and_plot_model(X, y):
  model = Sequential()
  model.add(Dense(4, input_shape=(2,), activation='tanh'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(SGD(lr=0.5),
                'binary_crossentropy',
                metrics=['accuracy'])
  model.fit(X, y, epochs=30, verbose=0);

  plot_decision_boundary(model, X, y)

X, y = make_circles(n_samples=1000,
                    noise=0.1,
                    factor=0.2,
                    random_state=0)

train_and_plot_model(X, y)

X, y = make_blobs(n_samples=1000,
                  centers=2,
                  random_state=0)

train_and_plot_model(X, y)

X, y = make_moons(n_samples=1000,
                  noise=0.1,
                  random_state=0)

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(SGD(lr=0.5),
              'binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, epochs=30, verbose=0);

train_and_plot_model(X, y)

# Useful links I used during this:
# https://docs.scipy.org/doc/numpy/user/quickstart.html
# https://ml-cheatsheet.readthedocs.io/en/latest/linear_algebra.html
