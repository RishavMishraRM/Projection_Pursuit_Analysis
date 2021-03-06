import numpy as np
import sys
sys.path.append("..")
from skpp import ProjectionPursuitRegressor
from matplotlib import pyplot as plt

X = np.arange(100).reshape(100, 1)
Y = (3*X**2 - 2*X)[:,0]/1000 + np.random.randn(100)

estimator = ProjectionPursuitRegressor()
estimator.fit(X, Y)

plt.scatter(X, Y, c='k', s=10, label='dummy data points')
plt.plot(estimator.predict(X), label='relationship fit')
plt.legend()

plt.title('The relationship found between input and output\nfor a ' +
	'one-dimensional example.')
plt.xlabel('Single-dimensional input X')
plt.ylabel('Single-dimensionsl output Y')
plt.show()