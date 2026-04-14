import numpy as np # linear algebra
import pandas as pd # Data processing, CSV file I/O
import matplotlib.pyplot as plt # Building diagrams
import seaborn as sns  # Visualization tool
import warnings # Manipulate with terminal calls

data = pd.read_csv('pokemon.csv') # write data from file

# CLEANING DATA
# DIAGNOSE DATA for CLEANING
"""
print(data.head()) # Виводяться перші рядки (за замовчуванням 5)
print(data.tail()) # Виводяться останні рядки (за замовчуванням 5)
print(data.columns) # Проводяться операції з стовбцями
print(data.shape) # Виводить кількість рядків та стовбців
print(data.info()) # Виводить інформацію DataFrame
"""
# EXPLORATORY DATA ANALYSIS

# print(data['Type 2'].value_counts(dropna =False)) # Повертає значення стовбця Type 2 без ігнорування NaN та None
# print(data.describe()) # Повертає статистичні дані

# VISUAL EXPLORATORY DATA ANALYSIS
data.boxplot(column='Attack',by = 'Legendary') # Розподіл даних
# plt.show()

# TIDY DATA

data_new = data.head()
# print(data_new)
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense']) # Переведення в інший формат таблиці
# print(melted)


# PIVOTING DATA
melted.pivot(index = 'Name', columns = 'variable',values='value')
# print(melted)

# CONCATENATING DATA
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True) # додає вертикально таблицю
# print (conc_data_row)

data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat([data1,data2], axis = 1) # додає горизонтально таблицю
# print (conc_data_col)

# DATA TYPES
"""
# print (data.dtypes)
data['Type 1'] = data['Type 1'].astype('category') # Зміна типу даних блоку
data['Speed'] = data['Speed'].astype('float')
# print (data.dtypes)
"""

# MISSING DATA and TESTING WITH ASSERT
# print(data.info())
# print(data["Type 2"].value_counts(dropna = False))
"""
data1 = data
print(data1["Type 2"].dropna(inplace = True)) # Виникає розсинхронізація даних
assert  data['Type 2'].notnull().all() # Нічого не вертає , бо nan значення були скинуті
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all() # Нічого не вертає , бо nan значення немає
"""

# PANDAS FOUNDATION
# BUILDING DATA FRAMES FROM SCRATCH
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col)) # Пакування списків
data_dict = dict(zipped) # Переведення списків до формату словника
df = pd.DataFrame(data_dict) # Переведення словника до формату DataFrame
# print(df)

df["capital"] = ["madrid","paris"] # Додавання нового стовбця та його даних
df["income"] = 0  # Додавання нового стовбця та автозаповнення даних
# print(df)

# VISUAL EXPLORATORY DATA ANALYSIS
"""
data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot() # дані на одному графіку
data1.plot(subplots = True) # на окремих
data1.plot(kind = "scatter",x="Attack",y = "Defense") # діаграма розсіювання
data1.plot(kind = "hist", y = "Defense", bins = 50,range = (0,250)) # гістограма
plt.show()

fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist", y = "Defense", bins = 50, range = (0,250), density = True, ax = axes[0]) # density = True (щільність імовірності)
data1.plot(kind = "hist", y = "Defense", bins = 50, range = (0,250), density = True, ax = axes[1], cumulative = True) # підсумовує імовірності
plt.savefig('graph.png')
plt.show()
"""

# STATISTICAL EXPLORATORY DATA ANALYSIS
# print(data.describe())

# INDEXING PANDAS TIME SERIES
time_list = ["1992-03-08","1992-04-12"] # Тип даних str
# print(type(time_list[1]))
datetime_object = pd.to_datetime(time_list) # Зміна типу даних до DatetimeIndex
# print(type(datetime_object))

warnings.filterwarnings("ignore")
data2 = data.head() # Взяті перші 5 стовбців
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object # Запис даних формату DatetimeIndex
data2 = data2.set_index("date") # Запис дати у вигляді індексу в DataFrame
# print(data2)
# print(data2.loc["1993-03-16"]) # Виконання операцій з індексуванням
# print(data2.loc["1992-03-10":"1993-03-16"])

# RESAMPLING PANDAS TIME SERIES
# print(data2.resample("A").mean(numeric_only=True)) # Індексує по рокам (mean видає середнє значення)
# print(data2.resample("M").mean(numeric_only=True)) # По місяцам (пусті місяці проставляють Nan)
# print(data2.resample("M").first().interpolate("linear")) # По місяцам (пусті місяці лінійно додають до числових значень від двох найближчих точок) (Закоментований блок DATA TYPES !!!)
# print(data2.resample("M").mean(numeric_only=True).interpolate("linear")) # Додається середне значення між двома найближчими точками