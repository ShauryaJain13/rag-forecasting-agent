import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


class NaiveModel:
    """
    This class is meant to serve as a baseline for the models.
    In this model, we will predict the next value using the previous value
    """
    name = "naive"

    def fit(self, series, covariates=None):
        """
        Stores the last observed value
        """
        self.last_value = series.iloc[-1]
        return self

    def predict(self, horizon, covariates=None):
        """
        Predicts the last value as the future predicted value
        """
        return np.repeat(self.last_value, horizon)


class HoltWinters:
    """
    This class is meant to fit and predict Holt-Winters' Exponential
    Smoothing
    """

    name = "holt_winters"

    def __init__(self, seasonality=7):
        self.seasonality = seasonality
        self.model = None
        self.fitted_model = None

    def fit(self, series, covariates=None):
        """
        Fitting Holt-Winters to the given series data
        """
        self.model = ExponentialSmoothing(series, trend="add", seasonal="add",
                                          seasonal_periods=self.seasonality)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, horizon, covariates=None):
        """
        Based on the fitted model, predicts the values for the horizon
        """
        return self.fitted_model.forecast(horizon)


class XGBoost:
    """
    This class is to fit and predict models to the XGBoost Model
    """
    name = "xgboost"

    def __init__(self, n_lags=7):
        self.lags = n_lags
        self.model = XGBRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.05,
                                  objective="reg:squarederror")
        self.history = None

    def _create_features(self, series, covariates=None):
        """
        Creating lag features for the regression-forecasting model
        """
        df = pd.DataFrame({"target": series})
        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = series.shift(lag)

        if covariates is not None:
            aligned_covariates = covariates.reindex(series.index)
            df = df.join(aligned_covariates)
            self.covariate_columns = list(aligned_covariates.columns)
        else:
            self.covariate_columns = []

        df = df.dropna()
        X = df.drop(columns="target")
        y = df["target"]
        return X, y

    def fit(self, series, covariates=None):
        """
        Fitting XGBoost to the given series data
        """
        X, y = self._create_features(series, covariates)
        self.model.fit(X, y)
        self.history = list(series)
        self.last_covariate_row = (covariates.iloc[-1]
                                   if covariates is not None else None)
        return self

    def predict(self, horizon, covariates=None):
        """
        Predicting the values for the given horizon
        """
        history = self.history.copy()
        predictions = []

        for step in range(horizon):
            # features = np.array(history[-self.lags:]).reshape(1, -1)
            # prediction = self.model.predict(features)[0]

            lag_values = np.array(history[-self.lags:])
            row = {f"lag_{i}": lag_values[-i] for i in range(1, self.lags + 1)}

            if covariates is not None and len(covariates) > step:
                cov_row = covariates.iloc[step]
            elif self.last_covariate_row is not None:
                cov_row = self.last_covariate_row
            else:
                cov_row = pd.Series(0, index=self.covariate_columns)

            for column in self.covariate_columns:
                row[column] = cov_row.get(column, 0)

            features = pd.DataFrame([row])

            # features = pd.DataFrame([lag_values], columns=[f"lag_{i}"
            #                                                for i in
            #                                                range(1, self.lags
            #                                                      + 1)])
            prediction = self.model.predict(features)[0]
            predictions.append(prediction)
            history.append(prediction)

        return np.array(predictions)


class SARIMAModel:
    """
    Seasonal ARIMA forecasting model.
    """

    name = "sarima"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def fit(self, series, covariates=None):
        """
        This function is to fit the data series to the model
        """
        self.model = SARIMAX(series, exog=covariates, order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)

        self.fitted_model = self.model.fit(disp=False)
        return self

    def predict(self, horizon, covariates=None):
        """
        This function is to predict the values of the series in the
        given horizon
        """
        if covariates is not None and len(covariates) < horizon:
            raise ValueError(
                "Not enough future covariate rows for the requested "
                "horizon.")
        return self.fitted_model.forecast(steps=horizon, exog=covariates)
