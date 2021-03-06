from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype = np.float64)
# ys = np.array([5,4,6,5,6,7], dtype = np.float64)

''' using
		hm -> no of data points to create`
		variance -> variance of dataset
		step -> how far on average to step up the y value
		correlation -> positive, negative or None ('pos', 'neg' or False)
'''
def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype = np.float64), np.array(ys, dtype=np.float64)
	
def best_fit_slope_and_intercept(xs,ys):
	m = (((mean(xs)*mean(ys))-mean(xs*ys))/
		(mean(xs)**2-mean(xs**2)))
	b = mean(ys) - m*mean(xs)
	return m,b

def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
	ys_mean_line = [ mean(ys_orig) for y in ys_orig]
	squared_error_mean = squared_error(ys_orig, ys_mean_line)
	squared_error_regr = squared_error(ys_orig, ys_line)
	return 1 - (squared_error_regr / squared_error_mean)

xs, ys = create_dataset(40, 40, 2, correlation = 'pos')

m,b = best_fit_slope_and_intercept(xs,ys)
print(m,b)

regression_line = [(m*x)+b for x in xs]

''' similar to
	for x in xs:
		regression_line.append(m*x+b)
'''

predict_x = 10
predict_y = m*predict_x + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# Scatterplot for x & y
plt.scatter(xs,ys)
# plot the best fit line
plt.plot(xs, regression_line)
plt.scatter(predict_x, predict_y, s=100, color='g')
plt.show()


''' Coefficient of determination
	How good of a fit is our best fit line?
		r^2 = 1 - (Square Error of best fit line (y-hat) ) / ( SE of mean of y's (y-bar) )
	higher r^2 => linear data, fits data appropriately
	lower r^2 => line doesn't fit the data that well..
	
		Why we use the square error? 
			-> to deal with positive values
			-> to deal with outliers
				(farther points get high squared error)
				standard is squared error but one can use
				 power to 4, 8, 16 etc..'''
