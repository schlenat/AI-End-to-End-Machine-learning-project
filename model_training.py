import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(path="used_cars.csv"):
    df = pd.read_csv(path)
    df['car_age'] = 2025 - df['year']
    df.drop(columns=['year'], inplace=True)
    df = pd.get_dummies(df, columns=['make', 'fuel', 'transmission'], drop_first=True)
    return df

def split_data(df):
    X = df.drop(columns=['price'])
    y = df['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    lr, rf = train_models(X_train, y_train)

    evaluate_model(lr, X_test, y_test, "Linear Regression")
    evaluate_model(rf, X_test, y_test, "Random Forest")

    save_model(rf, "random_forest_model.pkl")
