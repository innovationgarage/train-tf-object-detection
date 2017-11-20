import pandas as pd
import sys

df = pd.read_csv('labels/all_labels.csv')
valid_frac = float(sys.argv[1])
valid_size = int(valid_frac * df.filename.nunique())
train_size = df.filename.nunique() - valid_size
valid_df = df.loc[df.filename.isin(df.filename.sample(valid_size))]
train_df = df.loc[df.filename.isin(valid_df.filename.unique())==False]

train_df.to_csv('data/train_labels.csv', index_label=False, index=False)
valid_df.to_csv('data/test_labels.csv', index_label=False, index=False)

