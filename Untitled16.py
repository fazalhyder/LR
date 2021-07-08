#!/usr/bin/env python
# coding: utf-8

# In[210]:


import pandas as pd


# In[211]:


import matplotlib.pyplot as plt
import numpy as np


# In[212]:


df=pd.read_csv("kc_house_data.csv")


# In[213]:


df


# In[214]:


df.shape


# In[215]:


df.info()


# In[216]:


df['date'] = pd.to_datetime(df['date'])
df['Month'] = df['date'].apply(lambda date: date.month)
df['Year'] = df['date'].apply(lambda date: date.year)


# In[217]:


Y = df['price'].values


# In[218]:


print(X.shape)
print(Y.shape)


# In[219]:


df.head()


# In[220]:


df.isnull().sum()


# In[221]:


df.dropna(inplace=True)


# In[222]:


df.isnull().sum()


# In[223]:


sqft=df['sqft_living']


# In[224]:


sqft.plot()


# In[225]:


X = df[['bedrooms','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition',
       'grade','sqft_above','sqft_basement','sqft_living15','sqft_lot15']].values
y = df['price'].values


# In[226]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


# Normalize the data.

# In[227]:


std = StandardScaler()
X = std.fit_transform(X)


# In[228]:


from sklearn.ensemble import RandomForestRegressor


# In[229]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# In[230]:


std = StandardScaler()
X = std.fit_transform(X)


# In[231]:


rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(X_train,y_train)


# In[232]:


score_rfr = rfr.score(X_train,y_train)
prev_rfr = rfr.predict(X_test)
mae_rfr = mean_absolute_error(y_test,prev_rfr)
mse_rfr = mean_squared_error(y_test,prev_rfr)
rmse_rfr = np.sqrt(mean_squared_error(y_test,prev_rfr))


# In[233]:


print('Mae: ',mae_rfr)
print('Mse: ',mse_rfr)
print('Rmse: ',rmse_rfr)


# # LINEAR REGRESSION

# In[234]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[235]:


pred_lr = lr.predict(X_test)
score_lr = lr.score(X_train,y_train)


# In[236]:


mae_lr = mean_absolute_error(y_test,pred_lr)
mse_lr = mean_squared_error(y_test,pred_lr)
rmse_lr = np.sqrt(mse_lr)


# In[237]:


print('Mae_lr: ',mae_lr)
print('Mse_lr: ',mse_lr)
print('Rmse_lr: ',rmse_lr)


# In[238]:


import matplotlib.pyplot as plt


# In[239]:


def resizeplot(l,a):
    plt.figure(figsize=(l,a))


# In[240]:


resizeplot(10,6)
plt.scatter(y_test,pred_lr)
plt.plot(y_test,y_test,color='red')


# # example of plotting a gradient descent search on a one-dimensional function

# In[241]:



from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# derivative of objective function
def derivative(x):
	return x * 2.0

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
	# track all solutions
	solutions, scores = list(), list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# run the gradient descent
	for i in range(n_iter):
		# calculate gradient
		gradient = derivative(solution)
		# take a step
		solution = solution - step_size * gradient
		# evaluate candidate point
		solution_eval = objective(solution)
		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solutions, scores]

# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the solutions found
pyplot.plot(solutions, scores, '.-', color='red')
# show the plot
pyplot.show()


# In[ ]:





# In[ ]:




