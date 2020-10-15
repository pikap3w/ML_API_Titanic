# Import dependencies
import pandas as pd

# Load the dataset in a dataframe object and include only four features as mentioned
# url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
url = "/Users/pikap3w/Documents/Learning/Data_Science/ML_Python_Flask/titanic/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived']  # Only four features
df_ = df[include]

# Data Preprocessing
# Encode categorical values as numeric values (OHE), and replace missing values (NaNs) with 0
categoricals = []
for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)

# Create new column for every column/value combination, in a column_value format
df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Train ML Model
from sklearn.linear_model import LogisticRegression

dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

# Save the model (Serialization - aka pickling in Python)
# So it doesn't have to rerun every time you want to use it
import joblib

joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
# The logistic regression model is now persisted, and you can load it into memory
# Loading it back into your workspace is called Deserialization
lr = joblib.load('model.pkl')

# Save the data columns from training
# in case the incoming data set does not contain all possible values for the categorical variables
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Model columns dumped!")
