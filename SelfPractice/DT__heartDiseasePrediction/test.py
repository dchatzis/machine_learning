import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Dummy data
X = pd.DataFrame({
    'feature1': np.random.rand(400),
    'feature2': np.random.rand(400)
})
y = pd.Series(np.random.randint(0, 2, size=400))

# Train/val split
X_train_fit, X_train_eval = X.iloc[:320], X.iloc[320:]
y_train_fit, y_train_eval = y.iloc[:320], y.iloc[320:]

# Model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbosity=1,
    eval_metric='logloss'
)

# Fit with early stopping
xgb_model.fit(
    X_train_fit,
    y_train_fit,
    eval_set=[(X_train_eval, y_train_eval)],
    early_stopping_rounds=10,
    verbose=True
)