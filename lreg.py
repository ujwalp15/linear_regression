# import dependencies
# panda is an easy-to-use data structures and data analysis tools
import pandas as pa
from sklearn import linear_model
# matplotlib is a Python 2D plotting library
import matplotlib.pyplot as plot

# Read the data set
data_set = pa.read_fwf('age_bp.txt')
x_values = data_set[['Age']]
y_values = data_set[['BloodPressure']]

# Use linear regression to train the model
bp_prediction = linear_model.LinearRegression()
bp_prediction.fit(x_values, y_values)

# Plot the resultant values and visualize it
plot.scatter(x_values, y_values)
plot.plot(x_values, bp_prediction.predict(x_values))
plot.show()
