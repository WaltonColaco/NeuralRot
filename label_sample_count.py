import pandas as pd

def count_value_in_df(file_path, target_value):
    df = pd.read_csv(file_path)
    return (df.iloc[:,0] == target_value).sum()

file_path = 'gesture_data.csv'
target_value = 'six_seven'
print(count_value_in_df(file_path, target_value))
