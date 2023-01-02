# import required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# custom function to get common metrics in regression
from sklearn.metrics import mean_absolute_error


def linear_report (y_actual, y_prediction):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score

    mae = mean_absolute_error(y_actual, y_prediction)
    print('Mean Absolute Error is: ', mae)

    mse = mean_squared_error(y_actual, y_prediction)
    print ('Mean Squared Error is: ', mse)

    import math
    rmse = math.sqrt(mse)
    print('Root Mean Square Error is: ', rmse)

    r2 = r2_score(y_actual, y_prediction)
    print('R Squared Score is: ', r2)

    mape = 100 * (mae / y_actual)
    accuracy = 100 - np.mean(mape)
    print('Accuracy Score is: ', accuracy)

# import dataset as dataframe
df = pd.read_csv('stock-prediction-dataset.csv')
print(df.head())

plt.hist(df['current_price'])
plt.title('Current Price Distribution')
plt.xlabel('Current Price')
plt.ylabel('Number of Stocks')
print('Plot for Current Price across Number of Stocks')
plt.show()

plt.hist(np.log(df['current_price']))
plt.title('Current Price Distribution')
plt.xlabel('Log of Current Price')
plt.ylabel('Number of Stocks')
print('Plot for Log of Current Price across Number of Stocks')
plt.show()

cor_tab = df.corr()
cor_tab.style.background_gradient(cmap='coolwarm')

# drop the columns which are highly correlated to the target variable or with other features
drop_list = ['return_on_equity', 'high_price', 'low_price', 'ROE']
df.drop(drop_list, axis = 1, inplace=True)
print(df.head())

# delete target variable from dataframe and take log of target variable
target_log = np.log(df['current_price'])
target_org = df['current_price']
share_id = df['Unnamed: 0']

df.drop(['current_price', 'Unnamed: 0'], axis=1, inplace=True)

# ## Train the Model
from sklearn.model_selection import train_test_split
train_features, test_features, train_target, test_target = \
    train_test_split(df, target_log, test_size=0.10, random_state=77)

# ## Baseline Model
book_value_test = test_features['Book_Value']
# compute Mean Absolute Error
mae_baseline = mean_absolute_error(test_target, np.log(book_value_test))

linear_report(test_target, np.log(book_value_test))

# actual error in stock values
linear_report(np.exp(test_target), book_value_test)

# ## XGboost Model
# mean absolute error is 1.31; have to improve
import xgboost as xgb

# data preparation for xgb
dtrain = xgb.DMatrix(train_features, label=train_target)
dtest = xgb.DMatrix(test_features, label=test_target)
dwhole = xgb.DMatrix(df, label=target_log)

params = {
    'max_depth' : 6,
'min_child_weight': 13,
    'eta':.005,
    'subsample': 0.8,
    'colsample_bytree': 1,
    # other parameters
    'objective':'reg:squarederror',
    'eval_metric': "mae"
}

num_boost_round = 1500

model_xgb = xgb.train(params, dtrain, num_boost_round,
                      evals=[(dtest, "Test")], early_stopping_rounds=10)

fscores = model_xgb.get_fscore()

fscores = pd.DataFrame.from_dict(fscores, orient='index', columns=['Score'])
fscores = fscores['Score'].sort_values(ascending=False)
fscores[0:20]

predictions = model_xgb.predict(dtest)
linear_report(test_target, predictions)

# model doesn't tell if stock price will go up or down
# you can check if stock is overvalue or undervalue

predictions_vs_target = pd.DataFrame(columns=['Stock ID', 'Actual Price', 'Predicted Price', 'Difference'])
predictions_vs_target['Actual Price'] = np.exp(test_target)
predictions_vs_target['Predicted Price'] = np.exp(predictions)
predictions_vs_target['Difference'] = predictions_vs_target['Predicted Price'] \
                                      - predictions_vs_target['Actual Price']

list_stocks = []
for i in predictions_vs_target.index:
    x = share_id[i]
    list_stocks.append(x)

predictions_vs_target['Stock ID'] = list_stocks
predictions_vs_target['Predicted Price'] = predictions_vs_target['Predicted Price'].astype('float64')
predictions_vs_target['Verdict'] = predictions_vs_target['Difference'].apply(
    lambda x: 'Under Valued' if x >= 0 else 'Over Valued')
predictions_vs_target['Percentage Difference'] = \
    (predictions_vs_target['Difference'] / predictions_vs_target['Actual Price']) * 100

print(predictions_vs_target)

# ## Predctions on train dataset
predictions = model_xgb.predict(dtrain)

train_prediction = pd.DataFrame(columns=['Stock Id', 'Actual Price', 'Predicted Price', 'Difference'])
train_prediction['Actual Price'] = np.exp(train_target)
train_prediction['Predicted Price'] = np.exp(predictions)
train_prediction['Difference'] = train_prediction['Predicted Price'] - train_prediction['Actual Price']

list_stocks = []
for i in train_prediction.index:
    x = share_id[i]
    list_stocks.append(x)

train_prediction['Stock Id'] = list_stocks
train_prediction['Predicted Price'] = train_prediction['Predicted Price'].astype('float64')
train_prediction['Verdict'] = train_prediction['Difference'].apply(
    lambda x: 'Under Valued' if x >= 0 else 'Over Valued')
train_prediction['Percentage Difference'] = (train_prediction['Difference'] / train_prediction['Actual Price']) * 100

print(train_prediction)