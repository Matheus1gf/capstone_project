import numpy as np

def scale_data_test(test, scaler):
    """ Refactors data for testing

    Args:
    param test: Dataframe containing test data
    param scaler: predicted scale
    
    Returns:
    pred_test_inverted: Predetermination of treated tests
    """
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0:1].ravel()
    
    return X_test, y_test


def scaling(y, x, scaler, lstm=False):  
    """ Responsible for handling and determining out-of-scale data
    
    Args:
    param y: Dataframe with predictions
    param x: Dataframe with test
    param scaler: Scale data
    param lstm: Flag that determines whether the data to be scaled is LSTM or not

    Returns:
    pred_test_inverted: Inverted test data predictions
    """
    y = y.reshape(y.shape[0], 1, 1)
    
    if lstm == False:
        x = x.reshape(x.shape[0], 1, x.shape[1])
    
    pred_test = []
    for index in range(0,len(y)):
        pred_test.append(np.concatenate([y[index],x[index]],axis=1))
        
    pred_test = np.array(pred_test)
    pred_test = pred_test.reshape(pred_test.shape[0], pred_test.shape[2])
    
    pred_test_inverted = scaler.inverse_transform(pred_test)
    
    return pred_test_inverted