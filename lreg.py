import pandas as pa
from sklearn import linear_model
import matplotlib.pyplot as plot

data_set = pa.read_fwf('age_bp.txt')
x_values = data_set[['Age']]
y_values = data_set[['BloodPressure']]

bp_prediction = linear_model.LinearRegression()
bp_prediction.fit(x_values, y_values)

plot.scatter(x_values, y_values)
plot.plot(x_values, bp_prediction.predict(x_values))
plot.show()
