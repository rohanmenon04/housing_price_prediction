from train import TrainModel
from evaluate import EvaluateModel, evaluate

model = EvaluateModel('housing.csv')
evaluate(model=model, model_type='NN', model_path='models/neural_network/32_64_scaled_earlystop.keras', scaled=True)