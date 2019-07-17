import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor

# reading the data
file_path = <add_your_csv_file_here>
data = pd.read_csv(file_path)
print(data.columns)

# specifying which columns will we take in the model ... these are the "features" ...
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]
print(X.describe())

# top 5 rows
print(X.head())
# specifying what will we predict
y = data.Price

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we run this script. (random generator seed)
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Define model
# building the model ... here we care about which type we build through the import ...
model = DecisionTreeRegressor()

# Fit model
model.fit(train_X, train_y)

# and now .... voila .... predict !!
print(model.predict(X))

# measure the mean absolute error by comparing predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))







# compare same model with different parameters ... for example max leaf nodes in Decision tree ...
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))



# and then using the best value ... in this case it was 100 ...
final_model = DecisionTreeRegressor(max_leaf_nodes=100)
# fit the final model with the whole data for deployment ...
final_model.fit(X, y)
