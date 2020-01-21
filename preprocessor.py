import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


def get_processed_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # data exploration
    # print dataset information
    # print(train_df.info())
    # print(train_df.describe())
    # print(train_df.head(8))

    # we need to convert the features to the same scale
    # We need to convert the features to numerical features
    # We need to scope with missing values

    total = train_df.isnull().sum().sort_values(ascending=False)
    percent_1 = train_df.isnull().sum() / train_df.isnull().count() * 100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    print(missing_data.head(3))

    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    women = train_df[train_df['Sex'] == 'female']
    men = train_df[train_df['Sex'] == 'male']
    ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
    ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
    ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
    ax.legend()
    _ = ax.set_title('Male')
   # plt.show()


get_processed_data()
