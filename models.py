import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Function to train and save Logistic Regression model
def train_log_reg(X_train, y_train, X_train_scaled):
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)
    save_model(log_reg, 'log_reg_model.pkl')

# Function to train and save Random Forest model
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    save_model(rf, 'rf_model.pkl')

# Function to train and save SVM model
def train_svm(X_train, y_train, X_train_scaled):
    svm = SVC()
    svm.fit(X_train_scaled, y_train)
    save_model(svm, 'svm_model.pkl')

# Helper function to save model to disk
def save_model(model, filename):
    with open(f'{filename}', 'wb') as f:
        pickle.dump(model, f)

# Helper function to save scaler to disk
def save_scaler(scaler):
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Load saved models
def load_models():
    with open('log_reg_model.pkl', 'rb') as f:
        log_reg = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return log_reg, rf, svm, scaler
