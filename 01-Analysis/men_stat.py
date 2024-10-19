import pandas as pd

def men_stat(df: pd.DataFrame):
    ages = df[(df['Sex'] == 'male') & (df['Survived'] == 0)]['Age'].dropna()
    return ages.max(), ages.min(), ages.median(), ages.mean(), ages.var()
    
# ПРОВЕРКА:
data = pd.read_csv('data/titanic_train.csv')

######################################################
mx, mn, med, mean, var = men_stat(data.copy())
assert mean - 31.618 < 1e-3
assert med == 29
assert mx == 74
assert mn == 1
assert var - 197.571 < 1e-3
######################################################