import pandas as pd
from sklearn.model_selection import train_test_split



def batterSplit(df, target_col='', test_size=0.2):
    """
    Split the data into training and testing dataframes. instead of numpy arrays
    """
    X = df.drop(target_col, axis=1)
    y = df[[target_col]]  # This keeps 'RunTime ' as a DataFrame
    # save the columns names
    feature_names = X.columns
    target_col = y.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # turn X_train, X_test, y_train, y_test back to dataframes
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.DataFrame(y_train, columns=target_col)
    y_test = pd.DataFrame(y_test, columns=target_col)
    return X_train, X_test, y_train, y_test