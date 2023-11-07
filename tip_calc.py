# Machine learning model that predicts the amount a person will tip based on a range of factors

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("tips.csv")
print("Showing head data from 'tips.csv': ")
print(data.head())

"""Data Graphs and Charts showing relationship between tips and other variables"""

# Scatter Chart of tips based on day of the week
# figure1 = px.scatter(data_frame = data, x="total_bill",
#                    y="tip", size="size", color="day", trendline="ols")
# figure.show()

# Scatter Chart of tips based on gender of patron
# figure2 = px.scatter(data_frame = data, x="total_bill",
#                    y="tip", size="size", color="sex", trendline="ols")
# figure2.show()

# Scatter Chart of tips based on time of day
# figure3 = px.scatter(data_frame = data, x="total_bill",
#                    y="tip", size="size", color="time", trendline="ols")
# figure3.show()

# Pie chart of tips based on day of the week
# figure4 = px.pie(data, values="tip", names="day", hole = 0.5)
#
# figure4.show()

# Pie chart of tips based on gender of the patron
# figure5 = px.pie(data, values="tip", names="sex", hole = 0.5)
#
# figure5.show()

# Pie chart of tips based on whether the patron was a smoker or not
# figure6 = px.pie(data, values="tip", names="smoker", hole = 0.5)
#
# figure6.show()

# Pie chart of tips based on time of day
# figure7 = px.pie(data, values="tip", names="time", hole = 0.5)
#
# figure7.show()


"""TIP PREDICTION MODEL"""

data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
print("Mapping data for prediction...")
print(data.head())

x = np.array(data[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(data["tip"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)

# Features = [[total_bill, sex, smoker, day, time, size]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
print(f"Predicting tip size based on Total Bill, Sex, Smoker, Day, Time, and Size of Group ({features})...")
print(model.predict(features))
