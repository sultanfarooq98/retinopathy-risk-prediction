import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import train_log_reg, train_random_forest, train_svm, save_scaler

# Load the dataset
data = pd.read_csv('data.csv')

# Data Preprocessing
def preprocess_data(data):
    # Drop PatientID
    data = data.drop('PatientID', axis=1)

    # Split Blood_Pressure into two separate columns: Systolic and Diastolic
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood_Pressure'].str.split('/', expand=True)
    data = data.drop('Blood_Pressure', axis=1)

    # Convert Gender to binary (0 for Male, 1 for Female)
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Convert Systolic_BP and Diastolic_BP to numeric
    data['Systolic_BP'] = pd.to_numeric(data['Systolic_BP'], errors='coerce')
    data['Diastolic_BP'] = pd.to_numeric(data['Diastolic_BP'], errors='coerce')

    # Handle missing values by filling with the median
    data = data.fillna(data.median())

    return data

# Preprocess the data
data = preprocess_data(data)

# Split the data into features and target
X = data.drop('Retinopathy_Level', axis=1)
y = data['Retinopathy_Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features for models that require scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
save_scaler(scaler)

# Train and save Logistic Regression, Random Forest, and SVM models
train_log_reg(X_train, y_train, X_train_scaled)
train_random_forest(X_train, y_train)
train_svm(X_train, y_train, X_train_scaled)
