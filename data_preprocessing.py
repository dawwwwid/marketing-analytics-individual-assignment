import pandas as pd
import numpy as np

def Preprocess(train, test):
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)

    combine = [train_df, test_df]

    for df in combine:
        isAlone(df)
        drop_(df)
        fill_(df, train_df)
        map_(df)
        fill_age(df, train_df)
        age_banding(df)
        fare_banding(df)
    
    train_df.to_csv('../data/preprocessed_train.csv')
    test_df.to_csv('../data/preprocessed_test.csv')
    return train_df, test_df

def isAlone(df):
    df['isAlone'] = 1
    df.loc[df.SibSp + df.Parch > 0, 'isAlone'] = 0

def drop_(df):
    df.drop(['Ticket', 'Name', 'Cabin', 'Parch', 'SibSp'], inplace=True, axis=1)

def fill_(df, base_df):
    df.Embarked.fillna('S', inplace=True)
    df.Fare.fillna(base_df.Fare.median(), inplace=True)

def map_(df):
    df.Sex = df.Sex.map({
        'female': 1,
        'male': 0
    }).astype(int)
    df.Embarked = df.Embarked.map({
        'S': 0,
        'C': 1,
        'Q': 2
    }).astype(int)

def fill_age(df, base_df):
    n = base_df['Pclass'].nunique()
    m = base_df['Sex'].nunique()
    ages = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            guess = base_df[(base_df['Sex'] == i) & (base_df['Pclass'] == j + 1)]['Age'].dropna()
            ages[i, j] = guess.median()

    for i in range(m):
        for j in range(n):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), 'Age'] = ages[i, j]

def age_banding(df):
    df.loc[df.Age <= 16, 'Age'] = 0
    df.loc[(df.Age > 16) & (df.Age <= 32), 'Age'] = 1
    df.loc[(df.Age > 32) & (df.Age <= 48), 'Age'] = 2
    df.loc[(df.Age > 48) & (df.Age <= 64), 'Age'] = 3
    df.loc[df.Age > 64, 'Age'] = 4
    df.Age = df.Age.astype(int)

def fare_banding(df):
    df.loc[df.Fare <= 7.91, 'Fare'] = 0
    df.loc[(df.Fare > 7.91) & (df.Fare <= 14.454), 'Fare'] = 1
    df.loc[(df.Fare > 14.454) & (df.Fare <= 31), 'Fare'] = 2
    df.loc[df.Fare > 31, 'Fare'] = 3
    df.Fare = df.Fare.astype(int)
