import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("data/productivity_data.csv")

X = df[["Hours_Studied", "Sleep_Hours", "Phone_Usage"]]
y = df["Productivity_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained & saved successfully!")
print("Mean Absolute Error:", error)