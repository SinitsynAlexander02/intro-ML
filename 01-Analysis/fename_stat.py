import pandas as pd

def fename_stat(df: pd.DataFrame) -> pd.DataFrame:
  miss_names = df[df['Name'].str.contains('Miss')]['Name'].str.extract(r'Miss\.\s+([^\s\(]+)')[0]
  popularity = miss_names.value_counts().reset_index()
  popularity.columns = ['Name', 'Popularity']
  return popularity.sort_values(by=['Popularity', 'Name'], ascending=[False, True]).reset_index(drop=True)
  
# ПРОВЕРКА:
girls = pd.read_csv('data/popular_girls.csv')
data = pd.read_csv('data/titanic_train.csv')

######################################################
assert_frame_equal(
    fename_stat(data),
    girls
)
######################################################