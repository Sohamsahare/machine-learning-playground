import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

# API key for quandl data
quandl.ApiConfig.api_key = "Mz4exfd6X-aPx9y1Bf_3"
df = quandl.get('WIKI/GOOGL')

# Naming feautes of a data frame
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# Manipulating data features which had relationships between each other
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Redefining  features in the dataframe
df = df[['Adj. Close','HL_PCT','PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

# fields with NA (Not Available) are filled with -99999
df.fillna( -99999, inplace = True )

# Use 10% of the latest data to predict the outcome
# Ex-> last 10 days to predict today's closing price
forecast_out = int(math.ceil(0.1 * len(df)))

# Shifting the label column of each row to 10 days in future
# (whatever  time frame we defined at line 25)
df['label'] = df[forecast_col].shift(-forecast_out)

# Drop NA values
df.dropna(inplace = True)

# Features -> Caps X , Labels-> Lowercase y

# Features are every column except 'label'
X = np.array(df.drop(['label'],1))
# Labels are stored separately
y = np.array(df['label'])

# Scale features wrt to other training data 
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace = True)

y = np.array(df['label'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs = -1) # 97.77% accuracy
# clf = svm.SVR() -> SVM LinearRegression 75% accuracy
# clf = svm.SVR(kernel = 'poly')  69% accuracy
clf.fit(X_train,y_train)

# Saving the classifier
# with open('linearregression.pickle','wb') as f:
#	pickle.dump(clf,f)
	
# to open 
# pickle_in = open('linearregression.pickle','rb')
# clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set  = clf.predict(X_lately)

# print(forecast_set, accuracy, forecast_out)
print(accuracy)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# to name labels on x axis
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

# to plot the graph 
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

