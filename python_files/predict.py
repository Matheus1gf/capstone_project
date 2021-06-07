import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

model_scores = {}

def predict_data(unscaled_predictions, original_data, flag=True):
    """ Data prediction based on out-of-scale predictions and the original data
    Args:
    param unscaled_predictions: unscaled predictions returned by scaling function
    param original_data: original training data

    Returns:
    return: Dataframe with the sum of last year's sales and their monthly dates
    """
    result_list = []
    dates = list(original_data[-12:].date) 
    sales = list(original_data[-12:].sales)

    for x in range(0,len(sales)):
        result_dict = {}
        sum_predict = unscaled_predictions[x][0] + sales[x] if flag == True else unscaled_predictions[x] + sales[x]
        result_dict['predict_value'] = int(sum_predict)
        result_dict['date'] = dates[x]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)
    return df_result

def load_original():
    """ Loads training data from train.csv
    
    Args: None

    Return: Dataframe with data contained in data/train.csv
    """
    original = pd.read_csv('data/train.csv')  
    original = original.drop(columns = ['store', 'item'])
    original.date = pd.to_datetime(original.date, errors='coerce')
    original = original.groupby(pd.Grouper(key='date', freq='1M',axis='index')).sum()
    original = original.reset_index()
    original.date = original.date.dt.strftime("%Y-%m-01")
    original.date = pd.to_datetime(original.date, format='%Y-%m-%d', errors='coerce')

    return original

def plot_results(results, original_data, model_name):
    """ Prints the results in graphical format
    
    Args:
    param results: Dataframe with results data to be implemented
    param original_data: Dataframe with original data
    param model_name: Model to be implemented

    Return: None
    """
    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(original_data.date, original_data.sales, data=original_data, ax=ax, 
                 label='Original', color='mediumblue')
    sns.lineplot(results.date, results.predict_value, data=results, ax=ax, 
                 label='Predicted', color='Red')
    
    ax.set(xlabel = "Date",
           ylabel = "Sales",
           title = f"{model_name} Sales Forecasting Prediction")
    
    ax.legend()
    
    sns.despine()
    
    plt.savefig(f'model_output/{model_name}_forecast.png')
