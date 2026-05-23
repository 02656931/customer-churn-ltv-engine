import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('customer_data.csv')

# Features and target
X = data[['MonthlyCharges', 'Tenure']]
y = data['LifetimeValue']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict values
predictions = model.predict(X_test)

print("LTV Predictions:")
print(predictions[:5])
