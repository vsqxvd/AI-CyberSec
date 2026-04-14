import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

melbourne_data = pd.read_csv('melb_data.csv')
# print(melbourne_data.describe())
pd.set_option('display.max_columns', None) # Відображення усіх стовбців
# print(melbourne_data.head())
# print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0) # Відкидання Nan значень

y = melbourne_data.Price # Основне значення прогнозування
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] # Лист ознак прогнозування
X = melbourne_data[melbourne_features]
# print(X.describe())
# print(X.head())

melbourne_model = DecisionTreeRegressor(random_state=1) # Використання моделі дерева рішень
melbourne_model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are:")
# print(melbourne_model.predict(X.head()))
# print(melbourne_data.head())

predicted_home_prices = melbourne_model.predict(X)
# print("calculated mean absolute error:", mean_absolute_error(y, predicted_home_prices))

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y) # Тренування моделі
val_predictions = melbourne_model.predict(val_X) # Прогнозування
# print("calculated mean absolute error:", mean_absolute_error(val_y, val_predictions)) # Аналіз середньої похибки

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [50, 500, 790, 800]: # Цикл визначення найбільш вдачного значення max_leaf_nodes
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

forest_model = RandomForestRegressor(random_state=1) # Використання моделі випадкового лісу
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
# print("calculated mean absolute error:", mean_absolute_error(val_y, melb_preds))

