from sklearn.preprocessing import MinMaxScaler

def scale_data_train(train):
    """ Refactors data for training
    
    Args:
    param train: Dataframe with training data

    Returns:
    X_train: Scaled training dataset
    y_train: Scaled training dataset
    scaler: Scale data set
    """
    # Transforming the features on range -1 and 1
    # Refference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    
    # reshape training set
    # Refference https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0:1].ravel()
    
    return X_train, y_train, scaler

