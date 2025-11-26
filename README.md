# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the salary dataset into a Pandas DataFrame and inspect the first few rows using data.head().
2. Check the dataset for missing values using data.isnull().sum() and inspect the data structure using data.info().
3. Preprocess the categorical data. Use LabelEncoder to convert the "Position" column into numerical values.
4. Define the feature matrix (X) by selecting the relevant columns (e.g., Position, Level), and set the target variable (Y) as the "Salary" column.
5.  Split the dataset into training and testing sets using train_test_split()
6.  Initialize the Decision Tree Regressor and fit the model to the training data (x_train, y_train).
7.  Predict the target values on the testing set (x_test) using dt.predict().
8.  Calculate the Mean Squared Error (MSE) using metrics.mean_squared_error() and the R-squared score (r2_score()) to evaluate the model's performance.
9.  Use the trained model to predict the salary of an employee with specific input features (dt.predict([[5,6]])).

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ssnjana J 
RegisterNumber:  212224230240
*/
```

```
import pandas as pd

data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```


## Output:

<img width="414" height="315" alt="image" src="https://github.com/user-attachments/assets/91b34814-793e-4cca-89a4-fd624613e3be" />

mean squared error 

<img width="927" height="43" alt="image" src="https://github.com/user-attachments/assets/e919f5ac-9df7-4950-bf72-487f108e346b" />

r2

<img width="600" height="62" alt="image" src="https://github.com/user-attachments/assets/f2dc8722-fb2f-4987-b3fe-b03dcfa4def3" />

<img width="1286" height="155" alt="image" src="https://github.com/user-attachments/assets/518a73d3-3fc6-469a-ae4f-df1c5a8e51d1" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
