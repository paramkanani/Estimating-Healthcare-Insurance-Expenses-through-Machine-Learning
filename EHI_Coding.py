#Task-1
import pandas as pd

df = pd.read_csv ('./insurance.csv')

df.info()

#Task-2 

duplicates = df.duplicated().sum()

#Task-3
df.drop_duplicates (inplace=True)

#Task-4
null_values = df.isnull().sum()

#Task-5
from sklearn.preprocessing import LabelEncoder
lab_encode = LabelEncoder()

df['sex'] = lab_encode.fit_transform(df['sex'])
df['smoker'] = lab_encode.fit_transform(df['smoker'])

#Task-6
one_hot_encode = pd.get_dummies(df['region'])

#Task-7
df1 = pd.concat([df,one_hot_encode], axis=1)

#Task-8
df1.drop("region", axis=1, inplace=True)

#Task -9
from sklearn.model_selection import train_test_split

X = df1.drop("charges", axis=1)

y = df1["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#task-10
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
rand_forest_model = RandomForestRegressor(n_estimators=50, n_jobs=2, random_state=42)

neg_mse_scores = cross_val_score(rand_forest_model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-neg_mse_scores)
std = rmse_scores.std()

#Task-11
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Make sure to import RandomForestRegressor
from sklearn.model_selection import train_test_split


predictions = rand_forest_model.predict(X_test)

rounded_predictions = np.round(predictions, 2)

y_test_rounded = np.round(y_test.values[:10], 2)

compare = pd.DataFrame({'Actual Charges': y_test_rounded, 'Predicted Charges': rounded_predictions[:10]})
