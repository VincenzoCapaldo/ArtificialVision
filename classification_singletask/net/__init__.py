from .models import Backbone_nFC


def get_model(model_name, num_label):
    return Backbone_nFC(num_label, model_name)

