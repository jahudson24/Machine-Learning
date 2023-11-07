import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

"""Importing csv file as 'data' and checking file to see if any values are null"""
data = pd.read_csv("advertising.csv")
print("Head values of 'advertising.csv' file: ")
print(data.head())

print("Checking to see if any data from 'advertising.csv' is null: ")
print(data.isnull().sum())

"""Group of graphs showing the relationship between advertising dollars spend and the number of units sold"""

# # Figure for relationship between TV ads and units sold
# figure = px.scatter(data_frame=data, x="Sales",
#                     y="TV", size="TV", trendline="ols")
# figure.show()
#
# # Figure for relationship between newspaper ads and units sold
# figure = px.scatter(data_frame=data, x="Sales",
#                     y="Newspaper", size="Newspaper", trendline="ols")
# figure.show()
#
# # Figure to show relationship between radio ads and units sold
# figure = px.scatter(data_frame=data, x="Sales",
#                     y="Radio", size="Radio", trendline="ols")
# figure.show()

# Correlation of all columns and sales column
correlation = data.corr()
print("Correlation between all columns and the Sales column: ")
print(correlation["Sales"].sort_values(ascending=False))

"""Sales Predictor Model"""
x = np.array(data.drop(["Sales"], axis=1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)

# features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 39, 62]])
print(f"Printing the prediction of units sold based on TV, Radio, and "
      f"newspaper advertising dollars ({features})...")
print(model.predict(features))
