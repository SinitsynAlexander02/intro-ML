import pandas as pd

def ZOOtable(zoo: dict) -> pd.DataFrame:
  df = pd.DataFrame(zoo).T.apply(pd.to_numeric, errors='ignore').dropna(axis=1).reset_index().rename(columns={'index': 'Type'}).sort_values(by = 'Type').reset_index(drop=True)
  return df.reindex(sorted(df.columns), axis=1)
  
# ПРОВЕРКА:
from pandas.testing import assert_frame_equal

######################################################
ZOO = {
        'cat': {'color':'black', 'tail_len': 50.0, 'injured': False},
        'dog': {'age': 6, 'tail_len': 30.5, 'injured': True}
      }
answer = pd.DataFrame(
    {
     'Type':['cat', 'dog'],
     'injured':[False, True],
     'tail_len':[50.0, 30.5]
    }
)
df = ZOOtable(ZOO)

assert_frame_equal(
    df.reset_index(drop=True),
    answer
)
######################################################
ZOO = {
        'cat': {'color':'black'},
        'dog': {'age': 6}
      }
answer = pd.DataFrame(
    {
     'Type':['cat', 'dog']
    }
)

df = ZOOtable(ZOO)

assert_frame_equal(
    df.reset_index(drop=True),
    answer
)
######################################################
ZOO = {
        'fish': {'injured': False, 'color':'gold', 'tail_len': 0.5, 'age': 0.5},
        'cat':  {'age': 8, 'color':'black', 'tail_len': 50.0, 'injured': False},
        'dog':  {'color':'grey', 'age': 6, 'tail_len': 30.5, 'injured': True}
      }
answer = pd.DataFrame(
    {
     'Type':['cat', 'dog','fish'],
     'age':[8.0, 6.0, 0.5],
     'color':['black', 'grey', 'gold'],
     'injured':[False, True, False],
     'tail_len':[50.0, 30.5, 0.5]
    }
)
df = ZOOtable(ZOO)

assert_frame_equal(
    df.reset_index(drop=True),
    answer
)
######################################################
ZOO = {
        'cat':  {'age': 8, 'color':'black', 'tail_len': 50.0, 'injured': False},
      }
answer = pd.DataFrame(
    {
     'Type':['cat'],
     'age':[8],
     'color':['black'],
     'injured':[False],
     'tail_len':[50.0]
    }
)
df = ZOOtable(ZOO)

assert_frame_equal(
    df.reset_index(drop=True),
    answer
)
######################################################