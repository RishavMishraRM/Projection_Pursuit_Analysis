import numpy
import sys
sys.path.append("..")
from skpp import ProjectionPursuitRegressor

n = 1000
d = 4
p = 10

X = numpy.random.rand(n, p) - 0.5
Y = numpy.zeros((n, d))
for j in range(5):
	alpha = numpy.random.randn(p) # projection vector
	projection = numpy.dot(X, alpha)
	# Generate random polynomials with coefficients in [-100, 100]
	f = numpy.poly1d(numpy.random.randint(-100, 100,
		size=numpy.random.randint(3+1)))
	beta = numpy.random.randn(d) # expansion vector
	Y += numpy.outer(f(projection), beta)

print('Average magnitude of squared Y per element', numpy.sum(Y**2)/Y.size)

ppr = ProjectionPursuitRegressor(opt_level='low', show_plots=True, plot_epoch=1)
ppr.fit(X, Y)

Yhat = ppr.predict(X)
error = numpy.sum((Y - Yhat)**2)/Y.size
print('Average magnitude of squared difference between predicted and original Y', error)