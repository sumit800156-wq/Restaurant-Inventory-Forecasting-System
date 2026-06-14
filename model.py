import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dataset Load
data = pd.read_csv("dataset.csv")

# Input aur Output
X = data[["Day"]]
y = data["Sales"]

# Model Training
model = LinearRegression()
model.fit(X, y)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully!")