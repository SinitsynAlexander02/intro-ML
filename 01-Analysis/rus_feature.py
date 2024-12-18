import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def rus_feature(df: pd.DataFrame) -> pd.DataFrame:
    def parse_date(date_str):
      dates = date_str.split()
      months = {
          'января': 1, 'февраля': 2, 'марта': 3,
          'апреля': 4, 'мая': 5, 'июня': 6,
          'июля': 7, 'августа': 8, 'сентября': 9,
          'октября': 10, 'ноября': 11, 'декабря': 12
      }
      return datetime(
          year=int(dates[2]), 
          month=months[dates[1]], 
          day=int(dates[0]) 
      )
      
    df['Полных лет'] = df.apply(lambda row: relativedelta(parse_date(row['Дата смерти']), parse_date(row['Дата рождения'])).years, axis=1)
    return df
    
# ПРОВЕРКА:
from pandas.testing import assert_frame_equal

######################################################
names = pd.DataFrame({'Имя':['Никола Тесла', 'Альберт Эйнштейн'],
                   'Дата рождения':['10 июля 1856 г.','14 марта 1879 г.'],
                   'Дата смерти':  ['7 января 1943 г.', '18 апреля 1955 г.']})
answer = pd.DataFrame({'Имя':['Никола Тесла', 'Альберт Эйнштейн'],
                       'Дата рождения':['10 июля 1856 г.','14 марта 1879 г.'],
                       'Дата смерти':  ['7 января 1943 г.', '18 апреля 1955 г.'],
                       'Полных лет':[86, 76]})
assert_frame_equal(
    rus_feature(names),
    answer
)
######################################################
names = pd.DataFrame({'Имя':['Никола Тесла'],
                   'Дата рождения':['10 июля 1856 г.'],
                   'Дата смерти':  ['7 января 1857 г.']})
answer = pd.DataFrame({'Имя':['Никола Тесла'],
                       'Дата рождения':['10 июля 1856 г.'],
                       'Дата смерти':  ['7 января 1857 г.'],
                       'Полных лет':[0]})
assert_frame_equal(
    rus_feature(names),
    answer
)
######################################################
names = pd.DataFrame({'Имя':['Никола Тесла'],
                   'Дата рождения':['1 января 2000 г.'],
                   'Дата смерти':  ['31 декабря 2000 г.']})
answer = pd.DataFrame({'Имя':['Никола Тесла'],
                       'Дата рождения':['1 января 2000 г.'],
                       'Дата смерти':  ['31 декабря 2000 г.'],
                       'Полных лет':[0]})
assert_frame_equal(
    rus_feature(names),
    answer
)
######################################################
names = pd.DataFrame({'Имя':['Никола Тесла'],
                   'Дата рождения':['1 января 2000 г.'],
                   'Дата смерти':  ['1 января 2001 г.']})
answer = pd.DataFrame({'Имя':['Никола Тесла'],
                       'Дата рождения':['1 января 2000 г.'],
                       'Дата смерти':  ['1 января 2001 г.'],
                       'Полных лет':[1]})
assert_frame_equal(
    rus_feature(names),
    answer
)
######################################################
names = pd.DataFrame({'Имя':['Никола Тесла', 'Никола Тесла', 'Никола Тесла'],
                   'Дата рождения':['1 марта 1999 г.', '1 марта 1999 г.', '1 марта 1999 г.'],
                   'Дата смерти':  ['27 февраля 2001 г.', '28 февраля 2001 г.', '1 марта 2001 г.']})

answer = pd.DataFrame({'Имя':['Никола Тесла', 'Никола Тесла', 'Никола Тесла'],
                       'Дата рождения':['1 марта 1999 г.', '1 марта 1999 г.', '1 марта 1999 г.'],
                       'Дата смерти':  ['27 февраля 2001 г.', '28 февраля 2001 г.', '1 марта 2001 г.'],
                       'Полных лет':[1, 1, 2]})
assert_frame_equal(
    rus_feature(names),
    answer
)
######################################################
names = pd.DataFrame({'Имя':['Никола Тесла', 'Никола Тесла', 'Никола Тесла'],
                   'Дата рождения':['1 марта 1899 г.', '1 марта 1899 г.', '1 марта 1899 г.'],
                   'Дата смерти':  ['27 февраля 1901 г.', '28 февраля 1901 г.', '1 марта 1901 г.']})

answer = pd.DataFrame({'Имя':['Никола Тесла', 'Никола Тесла', 'Никола Тесла'],
                       'Дата рождения':['1 марта 1899 г.', '1 марта 1899 г.', '1 марта 1899 г.'],
                       'Дата смерти':  ['27 февраля 1901 г.', '28 февраля 1901 г.', '1 марта 1901 г.'],
                       'Полных лет':[1, 1, 2]})
assert_frame_equal(
    rus_feature(names),
    answer
)
######################################################