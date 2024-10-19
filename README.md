<details>
<summary><h1>Analysis</h1></summary>
<details>
<summary><h2>Ближайший элемент</h2></summary>
Реализуйте функцию, принимающую на вход непустой тензор (может быть многомерным) $X$ и некоторое число $a$ и возвращающую ближайший к числу элемент тензора. Если ближайших несколько - выведите минимальный из ближайших. (Вернуть нужно само число, а не индекс числа!)

### Sample
#### Input:
```python
X = np.array([[ 1,  2, 13],
              [15,  6,  8],
              [ 7, 18,  9]])
a = 7.2
```
#### Output:
```python
7
```
[Solution:](./Отборочный/MTS-строки.py)
```python
import numpy as np

def nearest_value(X: np.ndarray, a: float) -> float:
    return X[np.abs(X - a) == np.abs(X - a).min()].min()
```
</details>
<details>
<summary><h2>Сортировка чисел</h2></summary>
Дан одномерный массив целых чисел. Необходимо отсортировать в нем только числа, которые делятся на $2$. При этом начальный массив изменять нельзя.

### Sample
#### Input:
```python
A = np.array([43, 66, 34, 55, 78, 105, 2])
```
#### Output:
```python
array([ 43,   2,  34,  55,  66, 105,  78])
```
[Solution:](./Отборочный/MTS-строки.py)
```python
import numpy as np

def sort_evens(A: np.ndarray) -> np.ndarray:
    b = A.copy()
    b[b % 2 == 0] = np.sort(b[b % 2 == 0])
    return b
```
</details>
<details>
<summary><h2>Страшные маски</h2></summary>
Даны трехмерный тензор размерности $X(n, k, k)$, состоящий из $0$ или $1$, или $n$ картинок $k \times k$. Нужно применить к нему указанную маску размерности $(k, k)$: В случае, если биты в маске и картинке совпадают, то результирующий бит должен быть равен $0$, иначе $1$.

### Sample
#### Input:
```python
X = np.array([
              [[ 1, 0, 1],
               [ 1, 1, 1],
               [ 0, 0, 1]],
             
              [[ 1, 1, 1],
               [ 1, 1, 1],
               [ 1, 1, 1]]
            ])
mask = np.array([[1, 1, 0],
                 [1, 1, 0],
                 [1, 1, 0]])
```
#### Output:
```python
array([[[0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]],

       [[0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]]])
```
[Solution:](./Отборочный/MTS-строки.py)
```python
import numpy as np

def tensor_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(X == mask, 0, 1)
```
ИЛИ
```python
import numpy as np

def tensor_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(X, mask)
```
</details>
<details>
<summary><h2>Сумма цифр в массиве</h2></summary>
На вход подается `np.ndarray` c натуральными числами. Надо получить массив сумм цифр в этих числах.

### Sample
#### Input:
```python
a = np.array([1241, 354, 121])
```
#### Output:
```python
array([ 8, 12, 4])
```
[Solution:](./Отборочный/MTS-строки.py)
```python
import numpy as np

def num_sum(a: np.ndarray) -> np.ndarray:
  digits = np.array(list(''.join(np.char.mod('%d', a)))).astype(int)
  return np.add.reduceat(digits, np.r_[0, np.char.str_len(np.char.mod('%d', a)).cumsum()[:-1]])
```
</details>
<details>
<summary><h2>Чистка NaN-ов</h2></summary>
Одна из важных проблем данных - пустые значения. В *numpy* и *pandas* они обычно объявляются специальным типом ```np.nan```. В реальных задачах нам часто нужно что-то сделать с этими значениями. Например заменить на 0, среднее или медиану.

Реализуйте функцию, которая во входной вещественной матрице ```X``` находит все значения ```nan``` и заменяет их на **медиану** остальных элементов столбца. Если все элементы столбца матрицы ```nan```, то заполняем столбец нулями.

### Sample
#### Input:
```python
X = np.array([[np.nan,      4,  np.nan],
              [np.nan, np.nan,       8],
              [np.nan,      5,  np.nan]])
```
#### Output:
```python
array([[0. , 4. , 8. ],
       [0. , 4.5, 8. ],
       [0. , 5. , 8. ]])
```
[Solution:](./Отборочный/MTS-строки.py)
```python
import numpy as np

def replace_nans(X: np.ndarray) -> np.ndarray:
    Y = X.copy()
    m = np.nanmedian(Y, axis=0)
    m[np.isnan(m)] = 0
    Y[np.isnan(Y)] = np.take(m, np.where(np.isnan(Y))[1])
    return Y
```
</details>
<details>
<summary><h2>Бухгалтерия зоопарка</h2></summary>
Вам на вход подается словарь, где ключ - это тип животного, а значение - словарь с признаками этого животного, где ключ - тип признака, а значение - значение признака (Типичный json проще говоря). Наименования признаков животного - всегда строки. Значения признаков - любой из типов pandas.

Вам следует создать табличку, где по строчкам будут идти животные, а по колонкам - их признаки, которая удовлетворяет следующим условиям:

* Тип животного нужно выделить в отдельную колонку `Type`
* Строки отсортированы по типу животного в алфавитном порядке
* Колонки отсортированы в алфавитном порядке, кроме колонки `Type` - она первая
* Индексы строк - ряд натуральных чисел начиная с 0 без пропусков

Имейте в виду, что признаки у двух животных могут не совпадать, значит незаполненные данные нужно заполнить `Nan` значением.

Верните на выходе табличку(`DataFrame`), в которой отсутствуют Nan значения. При этом могут отсутствовать некоторые признаки, но животные должны присутствовать **все**. Изначальные типы значений из словаря: `int64`, `float64`, `bool` и.т.д. должны сохраниться и в конечной табличке, а не превратиться в `object`-ы. (От удаляемых признаков этого, очевидно, не требуется).

### Sample
#### Input:
```python
ZOO = {
        'cat':{'color':'black', 'tail_len': 50, 'injured': False}, 
        'dog':{'age': 6, 'tail_len': 30.5, 'injured': True}
      }
```
#### Output:

|  | Type | injured |tail_len |
|--|----|--------|-------|
|0 | cat |  False | 50.0 |
|1 | dog |  True  | 30.5  |

[Solution:](./Отборочный/MTS-строки.py)
```python
import numpy as np
import pandas as pd

def ZOOtable(zoo: dict) -> pd.DataFrame:
  df = pd.DataFrame(zoo).T.apply(pd.to_numeric, errors='ignore').dropna(axis=1).reset_index().rename(columns={'index': 'Type'}).sort_values(by = 'Type').reset_index(drop=True)
  return df.reindex(sorted(df.columns), axis=1)
```
</details>
</details>
<details>
<summary><h1>IntroML</h1></summary>
</details>
<details>
<summary><h1>Linear</h1></summary>
</details>
<details>
<summary><h1>Features</h1></summary>
</details>
<details>
<summary><h1>DTandRF</h1></summary>
</details>
<details>
<summary><h1>Boosting</h1></summary>
</details>
<details>
<summary><h1>NeuralNets</h1></summary>
</details>
<details>
<summary><h1>ConvNets</h1></summary>
</details>
<details>
<summary><h1>NLP</h1></summary>
</details>
<details>
<summary><h1>NLPNN</h1></summary>
</details>
<details>
<summary><h1>ClusterTS</h1></summary>
</details>
