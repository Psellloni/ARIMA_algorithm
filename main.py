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

def main():
     read_some_file()

if __name__ == '__main__':
    main()