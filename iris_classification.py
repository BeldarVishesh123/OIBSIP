import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
csv_path = "iris.csv"  # make sure name matches your file
df = pd.read_csv(csv_path)

# Drop "Id" column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Features and target
X = df.drop(columns=["Species"])
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# User input
print("\nEnter flower details to predict species:")

sl = float(input("Sepal Length (cm): "))
sw = float(input("Sepal Width (cm): "))
pl = float(input("Petal Length (cm): "))
pw = float(input("Petal Width (cm): "))

# Prepare user input as dataframe (with correct column names)
user_data = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)

# Scale and predict
user_data = scaler.transform(user_data)
prediction = model.predict(user_data)

print("\nPredicted Species:", prediction[0])
