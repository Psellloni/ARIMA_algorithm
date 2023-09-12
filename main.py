#libraries
import pandas as pd
import polars as pl
import os


def read_some_file():
    abs_path = input('Enter absolute path: ')

    file_name, file_type = os.path.splitext(abs_path)

    if file_type == '.csv':
        df = 'csv'

        #df = pd.DataFrame(data=pd.read_csv(abs_path))

        return df
    elif file_type == '.parquet':
        df = 'parquet'

        #data = pl.read_ipc(abs_path)
        #df = pd.DataFrame(data=data.to_pandas())

        return df

    return None


def train_and_test_split(df: pl.DataFrame):
    days = input('Enter a number of days to be predicted: ')

    train = df[:(len(df) - days)]
    test = df[(len(df) - days):]

    return train, test


def main():
     df = read_some_file()
     train, test = train_and_test_split(df)

if __name__ == '__main__':
    main()