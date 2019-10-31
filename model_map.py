from base_model.lenet5 import create_lenet5

MNIST_MODEL_MAP = {
    'lenet5': create_lenet5
}

DATASET_TO_MODEL_MAP = {
    'mnist': MNIST_MODEL_MAP
}

#   return the model creation function
def get_model_fn(dataset_name, model_name):
    return DATASET_TO_MODEL_MAP[dataset_name][model_name]
