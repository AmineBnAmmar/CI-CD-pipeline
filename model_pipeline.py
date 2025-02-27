# model_pipeline.py
# Function to load and preprocess data
def prepare_data():
    import pandas as pd
    from category_encoders import OneHotEncoder
    from sklearn.preprocessing import StandardScaler

    # Load datasets
    df_80 = pd.read_csv("churn-bigml-80.csv")
    df_20 = pd.read_csv("churn-bigml-20.csv")

    # Define categorical features
    categorical_features = ["State", "International plan", "Voice mail plan"]

    # Apply OneHotEncoding to categorical features
    encoder = OneHotEncoder(cols=categorical_features, use_cat_names=True)
    df_80_encoded = encoder.fit_transform(df_80[categorical_features])
    df_20_encoded = encoder.transform(df_20[categorical_features])

    # Drop original categorical columns and join encoded ones
    df_80 = df_80.drop(columns=categorical_features).join(df_80_encoded)
    df_20 = df_20.drop(columns=categorical_features).join(df_20_encoded)

    # Define feature matrix and target variable
    X_train, y_train = df_80.drop(columns=["Churn"]), df_80["Churn"]
    X_test, y_test = df_20.drop(columns=["Churn"]), df_20["Churn"]

    # Normalize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, encoder, scaler


# Function to train XGBoost model
def train_model(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    return xgb


# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return acc, report


# Function to save model and encoders
def save_model(model, encoder, scaler, filename="xgboost.pkl"):
    import joblib

    joblib.dump({"model": model, "encoder": encoder, "scaler": scaler}, filename)
    print("Model saved as", filename)


# Function to load model
def load_model(filename="xgboost.pkl"):
    import joblib

    data = joblib.load(filename)
    return data["model"], data["encoder"], data["scaler"]
