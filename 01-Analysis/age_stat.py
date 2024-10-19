import pandas as pd

def age_stat(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
      values='Age',
      index=['Sex', 'Pclass'],
      aggfunc='max'
    )

# ПРОВЕРКА:
data = pd.read_csv('data/titanic_train.csv')

######################################################
idx = pd.MultiIndex.from_product([['female', 'male'],
                                  np.arange(1,4)],
                                 names=['Sex', 'Pclass'])
col = ['Age']

ans = pd.DataFrame(np.array([63.0, 57.0, 63.0, 80.0, 70.0, 74.0]), idx, col)

assert_frame_equal(
    age_stat(data.copy()),
    ans
)
######################################################