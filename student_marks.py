import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load your dataset
df = pd.read_csv("student_scores.csv")

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("Dataset columns:", df.columns.tolist())
print("\nDataset:")
print(df.head())

# 2. Select input and output columns
X = df[["study_hours_per_day"]]
y = df["final_exam_score"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("\nMean Squared Error:",
      round(mean_squared_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# 6. Predict marks
hours = float(input("\nEnter study hours per day: "))

new_data = pd.DataFrame({
    "study_hours_per_day": [hours]
})

marks = model.predict(new_data)

print("Predicted final exam score:", round(marks[0], 2))

# 7. Plot the results
plt.scatter(
    df["study_hours_per_day"],
    y,
    label="Actual marks"
)

# Sort values so the prediction line is plotted correctly
sorted_df = df.sort_values("study_hours_per_day")

plt.plot(
    sorted_df["study_hours_per_day"],
    model.predict(sorted_df[["study_hours_per_day"]]),
    label="Prediction line"
)

plt.xlabel("Study Hours Per Day")
plt.ylabel("Final Exam Score")
plt.title("Student Marks Prediction")
plt.legend()
plt.grid(True)
plt.show()