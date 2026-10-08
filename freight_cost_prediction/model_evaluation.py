from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth = 5):
    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, max_depth=6):
    model = RandomForestRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_freight_regressor(X_train, y_train):
    # Change Classifier to Regressor
    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    }

    # Use default scoring for Regression (R-squared)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluating regression model and return metrics.
    """
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds) *100

    print(f"\n{model_name} Performance:")
    print(f"MAE  :  {mae:.2f}")
    print(f"RMSE :  {rmse:.2f}")
    print(f"R²   :  {r2:.2f}%")

    return {
        "model_name" : model_name,
        "mae" : mae,
        "rmse" : rmse,
        "r2" : r2
    }