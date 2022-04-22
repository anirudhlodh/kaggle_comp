import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# A new feature 'is_train' is created to simplify our work with the datasets.
train['is_train']  = True
test['is_train'] = False

# Both datasets are combined together in order to perform data preprocessing.
data_set = pd.concat([train, test], axis=0,sort=False)
data_set.reset_index(drop=True, inplace=True)

print(data_set.head(3))

#start here

data_set_predict = data_set.loc[data_set['is_train'] == False]
data_set_predict = data_set.drop(['ID','Electric Utility','Vehicle Location','DOL Vehicle ID','Legislative District','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Electric Vehicle Type'], axis=1)
data_set = data_set.loc[data_set['is_train'] == True]

X = data_set.drop(['ID','VIN (1-10)','County','City','State','ZIP Code','Model Year','Make','Model','Electric Vehicle Type','Clean Alternative Fuel Vehicle (CAFV) Eligibility','Electric Range','Base MSRP','Legislative District','DOL Vehicle ID','Vehicle Location','Electric Utility','Expected Price ($1k)'],axis=1)
Y = data_set['Expected Price ($1k)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

# The data is being normalized based on the training subset.
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
data_set_predict = scaler.transform(data_set_predict)

# The predictions are calculated on the basis of four different models.
model1 = XGBRegressor(learning_rate=0.1, subsample=0.71, random_state=4, n_estimators=500)
model1.fit(X_train,Y_train)
predictions1 = model1.predict(X_test)

model2 = LinearRegression()
model2.fit(X_train,Y_train)
predictions2 = model2.predict(X_test)

model3 = GradientBoostingRegressor(learning_rate=0.1, random_state=4)
model3.fit(X_train,Y_train)
predictions3 = model3.predict(X_test)

model4 = LGBMRegressor(learning_rate=0.1, random_state=4)
model4.fit(X_train,Y_train)
predictions4 = model4.predict(X_test)

# The final predictions are calculated based on the following ensemble of the models.
predictions = predictions1*0.3 + predictions2*0.2 + predictions3*0.15 + predictions4*0.35

# Mean absolute percentage error.
error = metrics.mean_absolute_percentage_error(Y_test, predictions)

# The predictions for Kaggle challenge are calculated following the same ensemble.
output_pred = model1.predict(data_set_predict)*0.3 + model2.predict(data_set_predict)*0.2 + model3.predict(data_set_predict)*0.15 + model4.predict(data_set_predict)*0.35

output = pd.DataFrame()
output['Id'] = test['ID']
output['Expected Price ($1k)'] = output_pred

output.to_csv('TEAM_GRAVITY_submission.csv',index=False)

plt.scatter(predictions, Y_test, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], '--r', linewidth=2)
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Electric vehicle Price Prediction')
plt.show()