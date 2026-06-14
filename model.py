import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Dataset Load
data = pd.read_csv("dataset.csv")

# Input aur Output
X = data[["Day", "Temperature", "Weekend"]]
y = data["Sales"]

# Model Training
model = LinearRegression()
model.fit(X, y)

# Accuracy Check
pred = model.predict(X)
score = r2_score(y, pred)

print("Accuracy Score:", round(score, 2))

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Trained Successfully!")
