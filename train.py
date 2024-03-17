import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import time
from typing import Optional, Dict

class TrainModel:
    """
    This class pre-processes, cleans the data and trains a number of models based on the users preference
    :param path_to_dataset: receives the path to the dataset
    """
    def __init__(self, path_to_dataset: str):
        self.dataset_path = path_to_dataset
        self.dataset = pd.read_csv(path_to_dataset)

    def pre_process(self, test_size: float=0.2, visualise_call: Optional[bool]=False) -> pd.DataFrame:
        """
        This method processes the data into a format that is suitable for training
        :param test_size: the size used to split the training and testing data
        :returns: dataframes for training as well as testing in the format (x_train, y_train, x_test, y_test)
        """
        self.dataset.dropna(inplace=True)

        self.dataset['total_rooms'] = np.log(self.dataset['total_rooms'] + 1)
        self.dataset['total_bedrooms'] = np.log(self.dataset['total_bedrooms'] + 1)
        self.dataset['population'] = np.log(self.dataset['population'] + 1)
        self.dataset['households'] = np.log(self.dataset['households'] + 1)

        self.dataset['bedroom_ratio'] = self.dataset['total_bedrooms'] / self.dataset['total_rooms']
        self.dataset['household_rooms'] = self.dataset['total_rooms'] / self.dataset['households']

        self.dataset = self.dataset.join(pd.get_dummies(self.dataset.ocean_proximity))
        self.dataset = self.dataset.drop(['ocean_proximity'], axis=1)

        if visualise_call == True:
            dataset = self.dataset
            return dataset
        
        x = self.dataset.drop(['median_house_value'], axis=1)
        y = self.dataset['median_house_value']
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        return x_train, y_train, x_test, y_test
    
    def visualise_data(self) -> None:
        """
        This method visualises the datasets, in three different charts: 1. histogram, correlation chart, heatmap
        """
        full_dataset = self.pre_process(visualise_call=True)
        full_dataset.hist(figsize=(15,8))
        plt.figure(figsize=(15,8))
        sns.heatmap(full_dataset.corr(), annot=True, cmap='YlGnBu')
        plt.figure(figsize=(15,8))
        sns.scatterplot(x='latitude', y='longitude', data=full_dataset, hue='median_house_value', palette='coolwarm')
        plt.show()
    
    def grid_search(self, model, param_grid: Dict, cv: Optional[int]=5, scoring: Optional[str]='neg_mean_squared_error', save_location: Optional[str]=None, test_size: Optional[float]=0.2):
        """
        Performs grid search cross-validation on an existing model to optimize hyperparameters.

        Args:
            model: The model object to perform grid search on.
            param_grid: A dictionary of parameters and their values to try.
            cv: Number of cross-validation folds (default: 5).
            scoring: Scoring metric for evaluating model performance (default: 'neg_mean_squared_error').
            save_location: Optional path to save the best model found during grid search.
        """
        data = self.pre_process(test_size=test_size)
        x_train = data[0]
        y_train = data[1]
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
        grid.fit(x_train, y_train)

        if save_location:
            dump(grid.best_estimator_, save_location)

        print(f"Best parameters: {grid.best_params_}")
        print(f"Best score: {grid.best_score_}")

    def train_linear_regression(self, model_location: str, test_size: float=0.2) -> None:
        """
        Trains a linear regression model on the dataset and saves it to the specified location
        :param model_location: location for where to dump the model
        :param test_size: default as 0.2, specifies the train test split for training and testing
        """
        if os.path.exists(model_location):
            print ('This model already exists')
            print ('Exiting Program...')
            return
        
        data = self.pre_process(test_size=test_size)
        x_train = data[0]
        y_train = data[1]

        model = LinearRegression()
        model.fit(x_train, y_train)
        dump(model, model_location)

    def train_decision_tree(self, model_location: str, test_size: float=0.2, grid_search: Optional[bool]=False, param_grid: Optional[Dict]=None) -> None:
        """
        This method trains a decision tree model on the given dataset
        :param model_location: this is the name and location of the model to be saved
        :param test_size: optional parameter, size used for train-test split, default set to 0.2
        """
        if os.path.exists(model_location):
            print ('This model already exists')
            print ('Exiting Program...')
            return

        data = self.pre_process(test_size=test_size)
        x_train = data[0]
        y_train = data[1]

        model = DecisionTreeRegressor()

        if grid_search:
            if param_grid is None:
                raise ValueError("You must provide a parameter grid to perform grid search")
            self.grid_search(model=model, param_grid=param_grid, test_size=test_size)
        model.fit(x_train, y_train)
        dump(model, model_location)
    
    def train_random_forest(self, model_location: str, n_estimators: Optional[int]=100, random_state: Optional[int]=42, test_size: Optional[float]=0.2, grid_search: Optional[bool]=False, param_grid: Optional[Dict]=None) -> None:
        """
        Trains and saves a random forest model on the given dataset
        :param model_location: the name and location for the model to be saved
        :param n_estimators: optional parameter for the number of decision trees to construct
        :param random_state: optional parameter depth of the trees to sample
        :param test_size: optional parameter for splitting of the dataset
        """
        if os.path.exists(model_location):
            print ('This model already exists')
            print ('Exiting Program...')
            return
        data = self.pre_process(test_size=test_size)
        x_train = data[0]
        y_train = data[1]
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        if grid_search:
            if param_grid is None:
                raise ValueError("You must provide a parameter grid to perform grid search")
            
        model.fit(x_train, y_train)

        dump(model, model_location)
    
    def train_svm(self, model_location: str, kernel: Optional[str]='linear', C: Optional[float]=1.0, test_size: Optional[float]=0.2):
        """
        Trains and saves a SVM(regressor) model on the given dataset
        :param model_location: this is the name and location to store the trained model
        :param kernel: this is the kernel type to be used for the algorithm, default to linear
        :param C: regularization parameter, default to 1.0
        :param test_size: optional parameter for creating train/test splits
        """
        if os.path.exists(model_location):
            print ('This model already exists')
            print ('Exiting Program...')
            return
        data = self.pre_process(test_size=test_size)
        x_train = data[0]
        y_train = data[1]
        model = SVR(kernel=kernel, C=C)
        model.fit(x_train, y_train)

        dump(model, model_location)
    
    def train_gbm(self, model_location: str, loss: Optional[str]='squared_error', learning_rate: Optional[float]=0.1, n_estimators: Optional[int]=100, max_depth: Optional[int]=3, test_size: Optional[float]=0.2):
        """
        Trains and saves a gradient boosting machine (regressor) on the given dataset
        :param model_location: path to model to be saved after training
        :param loss: loss function to be monitored during gradient descent, default to squared error. Can also be 'absolute_error', 'huber', 'quantile'
        :param learning_rate: learning rate during optimization, default to 0.1
        :param n_estimators: optional parameter for the number of decision trees to construct, default to 100
        :param max_depth: depth to traverse each tree, default to 3
        :param test_size: size for train/test split, default at 0.2
        """
        if os.path.exists(model_location):
            print ('This model already exists')
            print ('Exiting Program...')
            return
        data = self.pre_process(test_size=test_size)
        x_train = data[0]
        y_train = data[1]
        model = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
        model.fit(x_train, y_train)

        dump(model, model_location)
    def train_neural_network(self, model_location: str, test_size: Optional[float]=0.2, epochs: Optional[int]=10, early_stopping: Optional[bool]=False,
                             monitor: Optional[str]=None, min_delta=None, patience=None, overwrite: Optional[bool]=False):
        """
        Trains a neural network model at a given location, includes optional parameters for early stopping as well as all early_stop criteria
        :param model_location: path to model to be saved after training
        :param test_size: optional parameter for creating train/test splits
        :param epochs: number of epochs for training, default set to 10
        :param early_stopping: boolean value for whether the model should implement early stopping or not
        :param monitor: metric to measure for early stopping
        :param min_delta: condition to stop training
        :param patience: number of epochs to wait during early stopping training
        :param overwrite: whether or not to overwrite existing model
        """
        if overwrite == False:
            if os.path.exists(model_location):
                print ('This model already exists')
                print ('Exiting Program...')
                return
        else:
            data = self.pre_process(test_size=test_size)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(data[0])
            y_train = data[1]
            model = Sequential([
                Input(shape=(x_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
            start = time.time()
            if early_stopping == False:
                model.fit(x_train, y_train, epochs=epochs)
            else:
                stopping = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, mode='auto', restore_best_weights=True)
                model.fit(x_train, y_train, epochs=epochs, callbacks=stopping)
            end = time.time()
            model.save(model_location, overwrite=True)
            print (f'Model took {(end-start)/60} minutes to train')