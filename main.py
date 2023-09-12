#libraries
import pandas as pd
import polars as pl
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
import pmdarima as pm


def read_some_file():
    file = input('Enter absolute path: ')

    file_name, file_type = os.path.splitext(file)

    #reading file and writing data to dataframe
    if file_type == '.csv':
        data = pd.read_csv(file)
        df = pd.DataFrame(data)

        return df
    elif file_type == '.parquet':
        data = pl.read_ipc(file)
        df = pd.DataFrame(data=data.to_pandas())

        return df

    return None

def test_model(df: pd.DataFrame):
    if df.shape[1] == 1:
        train, test = train_and_test_split(df)

        #building an arima model and making a prediction for one variable
        model = pm.auto_arima(train)
        prediction = model.predict(n_periods=len(test))
        mapa = mape(test, prediction)

        print(f'mape: {mapa}')
    else:
        #choosing column to be predicted
        print(df.columns)
        col_name = input(f'choose column to be predicted: ')

        train, test = train_and_test_split(df)

        # building a linear regression model and making a prediction
        model = LinearRegression()
        model.fit(train.drop(col_name, axis=1), train[col_name])
        prediction = model.predict(test.drop(col_name, axis=1))

        mapa = mape(test[col_name], prediction)

        # building an arima model and making a prediction
        model2 = pm.auto_arima(train[col_name], train.drop(col_name, axis=1))
        prediction2 = model2.predict(n_periods=len(test), X=test.drop(col_name, axis=1))

        mapa2 = mape(test[col_name], prediction2)

        if mapa < mapa2:
            print(f'LinearRegression mape: {mapa}')
        else:
            print(f'ARIMA mape: {mapa2}')

def train_and_test_split(df: pl.DataFrame):
    n_values = int(input('Enter a number of values to be predicted: '))

    #spliting dataframe to test and train
    train = df[:(len(df) - n_values)]
    test = df[(len(df) - n_values):]

    return train, test

def main():
     df = read_some_file()
     test_model(df)

if __name__ == '__main__':
    main()