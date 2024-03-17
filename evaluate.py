from train import TrainModel
from joblib import load
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict

class EvaluateModel:
    def __init__(self, dataset_path, test_size=0.2):
        self.model = TrainModel(dataset_path)
        self.x_train, self.y_train, self.x_test, self.y_test = self.model.pre_process(test_size=test_size)
    
    def linear_regression(self, path: str) -> Tuple[float, float]:
        """
        This method evaluates the linear regression model and prints statistics relating to the mean absolute error and mean percentage error
        :param path: this is the path to the linear regression model
        :returns: the mean absolute error and percentage error of the model
        """
        model = load(path)
        predictions = model.predict(self.x_test)
        delta = 0
        percentage_error = 0
        length = len(predictions)
        for prediction in zip(predictions, self.y_test):
            error = prediction[0] - prediction[1]
            delta += abs(error)
            if prediction[1] != 0:
                percentage_error += (error/prediction[1])
        
        delta /= length
        percentage_error /= length

        return delta, percentage_error
    
    def decision_tree(self, path: str) -> Tuple[float, float]:
        """
        This method evaluates a decision tree model and prints statistics relating to the mean absolute error and mean percentage error
        :param path: this is the path to the decision tree model
        :returns: the mean absolute error and percentage error of the model
        """
        model = load(path)
        predictions = model.predict(self.x_test)
        delta = 0
        percentage_error = 0
        length = len(predictions)
        for prediction in zip(predictions, self.y_test):
            error = prediction[0] - prediction[1]
            delta += abs(error)
            if prediction[1] != 0:
                percentage_error += (error/prediction[1])
        
        delta /= length
        percentage_error /= length

        return delta, percentage_error
    
    def random_forest(self, path: str) -> Tuple[float, float]:
        """
        This method evaluates a random model and prints statistics relating to the mean absolute error and mean percentage error
        :param path: this is the path to the random forest model
        :returns: the mean absolute error and percentage error of the model
        """
        model = load(path)
        predictions = model.predict(self.x_test)
        delta = 0
        percentage_error = 0
        length = len(predictions)
        for prediction in zip(predictions, self.y_test):
            error = prediction[0] - prediction[1]
            delta += abs(error)
            if prediction[1] != 0:
                percentage_error += (error/prediction[1])
        
        delta /= length
        percentage_error /= length

        return delta, percentage_error
    
    def svm(self, path: str) -> Tuple[float, float]:
        """
        This method evaluates a trained svm model and prints statistics relating to the mean absolute error and mean percentage error
        :param path: this is the path to the svm model
        :returns: the mean absolute error and percentage error of the model
        """
        model = load(path)
        predictions = model.predict(self.x_test)
        delta = 0
        percentage_error = 0
        length = len(predictions)
        for prediction in zip(predictions, self.y_test):
            error = prediction[0] - prediction[1]
            delta += abs(error)
            if prediction[1] != 0:
                percentage_error += (error/prediction[1])
        
        delta /= length
        percentage_error /= length

        return delta, percentage_error

    def gbm(self, path: str) -> Tuple[float, float]:
        """
        This method evaluates a trained gradient boosting machine model and prints statistics relating to the mean absolute error and mean percentage error
        :param path: this is the path to the svm model
        :returns: the mean absolute error and percentage error of the model
        """
        model = load(path)
        predictions = model.predict(self.x_test)
        delta = 0
        percentage_error = 0
        length = len(predictions)
        for prediction in zip(predictions, self.y_test):
            error = prediction[0] - prediction[1]
            delta += abs(error)
            if prediction[1] != 0:
                percentage_error += (error/prediction[1])
        
        delta /= length
        percentage_error /= length

        return delta, percentage_error

    def neural_network(self, path: str, scaled: Optional[bool]=False, verbosity: int=0) -> Tuple[float, float]:
        """
        This method evaluates a neural network model and prints statistics relating to the mean absolute error and mean percentage error

        Args:
            path: this is the path to the neural network model
            scaled: optional parameter representing if the inputs to the network need to be scaled or not
            verbosity: optional parameter for the verbosity level for the predictions of the network, default set at 0
        Returns:
            the mean absolute error and percentage error of the model
        """
        model = load_model(path)
        if scaled == True:
            scaler = StandardScaler()
            predictions = model.predict(scaler.fit_transform(self.x_test), verbose=verbosity)
        else:
            predictions = model.predict(self.x_test, verbose=verbosity)
        delta = 0
        percentage_error = 0
        length = len(predictions)
        for prediction in zip(predictions, self.y_test):
            error = prediction[0][0] - prediction[1]
            delta += abs(error)
            if prediction[1] != 0: 
                percentage_error += (error/prediction[1])
        
        delta /= length
        percentage_error /= length

        return delta, percentage_error

def evaluate(model: EvaluateModel, model_type: str, model_path: str, scaled: Optional[bool]=False, verbosity: int=0) -> None:
    """
    This function evaluates different models by calling the respective methods of the EvaluateModel class
    
    Args:
        model_type: a string representing the abbreviation for the model type eg. 'LR' for linear regression
        model_path: this is the path to the model to be evaluated
        scaled: this is an optional parameter which specifies if the neural network needs its inputs to be scaled or not
        verbosity: only relevant for NN model, determines the verbosity for when the predict method is called
    """
    if model_type.upper() == 'LR':
        delta, percentage_error = model.linear_regression(model_path)
        print (f'The linear regression model stored at {model_path} has a mean absolute error of: {delta:.2f}')
        print (f'The linear regression model stored at {model_path} has a percentage error of {(percentage_error*100):.2f}%')
    elif model_type.upper() == 'DT':
        delta, percentage_error = model.decision_tree(model_path)
        print (f'The decision tree model stored at {model_path} has a mean absolute error of: {delta:.2f}')
        print (f'The decision tree model stored at {model_path} has a percentage error of {(percentage_error*100):.2f}%')
    elif model_type.upper() == 'RF':
        delta, percentage_error = model.random_forest(model_path)
        print (f'The random forest model stored at {model_path} has a mean absolute error of: {delta:.2f}')
        print (f'The random forest model stored at {model_path} has a percentage error of {(percentage_error*100):.2f}%')
    elif model_type.upper() == 'SVM':
        delta, percentage_error = model.svm(model_path)
        print (f'The SVM model stored at {model_path} has a mean absolute error of: {delta:.2f}')
        print (f'The SVM model stored at {model_path} has a percentage error of {(percentage_error*100):.2f}%')
    elif model_type.upper() == 'GBM':
        delta, percentage_error = model.gbm(model_path)
        print (f'The GBM model stored at {model_path} has a mean absolute error of: {delta:.2f}')
        print (f'The GBM model stored at {model_path} has a percentage error of {(percentage_error*100):.2f}%')
    elif model_type.upper() == 'NN':
        delta, percentage_error = model.neural_network(model_path, scaled=scaled, verbosity=verbosity)
        print (f'The neural network model stored at {model_path} has a mean absolute error of: {delta:.2f}')
        print (f'The neural network model stored at {model_path} has a percentage error of {(percentage_error*100):.2f}%')
    else:
        print ("Your chosen model type doesn't not exist")