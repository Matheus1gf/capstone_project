import statsmodels.api as sm
from python_files.predict import *
from python_files.test import *
from python_files.train import *

model_scores = {}

def get_scores(original_data, model_name, unscaled_df = None):
    """ Take the quadratic error of last year datas forecast and difference
    
    Args: 
    param unscaled_df: Data frame out of scale
    param original_data: Dataframe with original data
    param model_name: Model name

    Return: None
    """
    if model_name == 'arima' and unscaled_df == None:
        # Refference https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        rmse = np.sqrt(mean_squared_error(original_data.sales_diff[-12:], original_data.forecast[-12:]))
        # Take the loss of mean absolute error regression of last year datas
        # Refference https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
        mean = mean_absolute_error(original_data.sales_diff[-12:], original_data.forecast[-12:])
        # Take the regression score of last year datas
        # Refference https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        r2 = r2_score(original_data.sales_diff[-12:], original_data.forecast[-12:])
    else:
        rmse = np.sqrt(mean_squared_error(original_data.sales[-12:], unscaled_df.predict_value[-12:])) 
        mean = mean_absolute_error(original_data.sales[-12:], unscaled_df.predict_value[-12:])
        r2 = r2_score(original_data.sales[-12:], unscaled_df.predict_value[-12:])

    model_scores[model_name] = [rmse, mean, r2]

    print(f"RMSE: {rmse}")
    print(f"Men Absolute Error: {mean}")
    print(f"R2 Score: {r2}")
    
def arima_model(data):
    """ Runs the SARIMAX model
    
    Args:
    param data: Dataframe containing the original data

    Return:
    seasonal_ar: Seasonal AutoRegressive Integrated Moving Average
    date: Date converted with sales_diff and forecast
    Predict: Predict with start in 40, end in 100 and dynamic with true
    """
    # Take the Seasonal AutoRegressive Integrated Moving Average
    # order: Order of Ar for 12 datas (one year)
    # seasonal_order: one interaction process and 12 years periodic
    # trend: constant
    # Refference https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    seasonal_ar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(12,0,0), seasonal_order=(0,1,0,12), trend='c').fit()

    # Predictions
    start, end, dynamic = 40, 100, True
    # Use the predict method with start in 40, ende in 100 and dynamic with true for usage the predicts in place of lagged dependent variables
    # Refference https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.predict.html
    data['forecast'] = seasonal_ar.predict(start=start, end=end, dynamic=dynamic) 
    predict = data.forecast[start+dynamic:end]
    
    data[['sales_diff', 'forecast']].plot(color=['mediumblue', 'Red'])
    
    get_scores(data, 'arima')

    return seasonal_ar, data, predict


def run_model(train_data, test_data, model, model_name):
    """ Responsible for running the models, regardless of type
    
    Args:
    param train_data: training dataframe
    param test_data: test dataframe
    param model: Tiop of the model to run
    param model_name: Name of the model to run

    Return: None
    """
    X_train, y_train, scaler_object = scale_data_train(train_data)
    X_test, y_test = scale_data_test(test_data, scaler_object)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Undo scaling to compare predictions against original data
    original_data = load_original()
    unscaled = scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_data(unscaled, original_data)
      
    get_scores(original_data, model_name, unscaled_df)
    
    plot_results(unscaled_df, original_data, model_name)