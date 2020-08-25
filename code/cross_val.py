import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


def cross_val(x, y, ridge_a=0.001, lasso_a=0.001):
    """
    Cross validation for a simple linear regression, lasso and ridge model at once
    """
    kf = KFold(n_splits=5, shuffle=True)
    lr_train_r2, lr_valid_r2 = [], []
    ridge_train_r2, ridge_valid_r2 = [], []
    lasso_train_r2, lasso_valid_r2 = [], []

    for train_ind, val_ind in kf.split(x, y):

        x_train, y_train = x[train_ind], y[train_ind]
        x_val, y_val = x[val_ind], y[val_ind]

        # simple linear regression
        lm = LinearRegression()

        lm.fit(x_train, y_train)
        lr_valid_r2.append(lm.score(x_val, y_val))
        lr_train_r2.append(lm.score(x_train, y_train))

        # ridge with feature scaling
        lm_reg = Ridge(alpha=ridge_a)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)

        lm_reg.fit(x_train_scaled, y_train)
        ridge_valid_r2.append(lm_reg.score(x_val_scaled, y_val))
        ridge_train_r2.append(lm_reg.score(x_val_scaled, y_val))

        # lasso with feature scaling (performed above in ridge)
        lm_las = Lasso(alpha=lasso_a)

        lm_las.fit(x_train_scaled, y_train)
        lasso_valid_r2.append(lm_las.score(x_val_scaled, y_val))
        lasso_train_r2.append(lm_las.score(x_val_scaled, y_val))

    print(f'Simple mean cv r^2 train: {np.mean(lr_train_r2):.3f} +- {np.std(lr_train_r2):.3f}')
    print(f'Simple mean cv r^2 valid: {np.mean(lr_valid_r2):.3f} +- {np.std(lr_valid_r2):.3f}')
    print('\n')
    print(f'Ridge mean cv r^2 train: {np.mean(ridge_train_r2):.3f} +- {np.std(ridge_train_r2):.3f}')
    print(f'Ridge mean cv r^2 valid: {np.mean(ridge_valid_r2):.3f} +- {np.std(ridge_valid_r2):.3f}')
    print('\n')
    print(f'Lasso mean cv r^2 train: {np.mean(lasso_train_r2):.3f} +- {np.std(lasso_train_r2):.3f}')
    print(f'Lasso mean cv r^2 valid: {np.mean(lasso_valid_r2):.3f} +- {np.std(lasso_valid_r2):.3f}')


def cross_val_simple(x, y):
    """
    Cross validation for a simple linear regression
    """
    kf = KFold(n_splits=5, shuffle=True)
    lr_train_r2, lr_valid_r2 = [], []
    rmse = []

    for train_ind, val_ind in kf.split(x, y):

        x_train, y_train = x[train_ind], y[train_ind]
        x_val, y_val = x[val_ind], y[val_ind]

        # simple linear regression
        lm = LinearRegression()
        lm.fit(x_train, y_train)
        lr_valid_r2.append(lm.score(x_val, y_val))
        lr_train_r2.append(lm.score(x_train, y_train))
        rmse.append(sqrt(mean_squared_error(y_val, lm.predict(x_val))))

    print(f'r^2 train: {np.mean(lr_train_r2):.3f} +- {np.std(lr_train_r2):.3f}')
    print(f'r^2 valid: {np.mean(lr_valid_r2):.3f} +- {np.std(lr_valid_r2):.3f}')
    print(f'rmse valid: {np.mean(rmse):.3f} +- {np.std(rmse):.3f}')
