# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Split the dataset into features (X = Hours Studied) and target (y = Marks Scored), then divide it into training (80%) and testing (20%) sets using train_test_split.
2.Model Training Fit a Linear Regression model on the training data. The model learns two parameters — the intercept (b0) and slope (b1) — by minimizing the sum of squared errors between actual and predicted values, forming the equation: Marks = b0 + b1 × Hours.

3.Model Evaluation Use the trained model to predict marks on the test set, then evaluate performance using Mean Squared Error (MSE) and R² Score to measure how well the regression line fits the data.

4.Prediction & Visualization Predict marks for a new input (e.g., 7.5 hours) using the learned equation, and plot the actual data points alongside the regression line to visually confirm the model's fit.

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored':  [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)
X = df[['Hours_Studied']]  
y = df['Marks_Scored']      

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Hours vs Marks')
plt.legend()
plt.show()
hours = int(input("\nEnter number of study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for studying {hours} hours = {predicted_marks[0]:.2f}")

```

## Output:
<img width="785" height="398" alt="Screenshot 2026-05-16 082454" src="https://github.com/user-attachments/assets/9c326760-c391-4b67-9811-727f0e103089" />
<img width="794" height="646" alt="Screenshot 2026-05-16 082553" src="https://github.com/user-attachments/assets/22613edc-5c50-4eb5-b7c1-c3c64eb89746" />






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
