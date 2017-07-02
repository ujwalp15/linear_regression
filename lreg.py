import pandas as pa
from sklearn import linear_model

data_set = pa.read_fwf('age_bp.txt')
x_values = data_set[['Age']]
y_values = data_set[['BloodPressure']]

bp_prediction = linear_model.LinearRegression()
bp_prediction.fit(x_values, y_values)
