'''
@Project : RepresentationLearningLand 
@File    : builder.py
@Author  : Wenyuan Li
@Date    : 2020/10/9 10:55 
'''
from .classification.classification_nets import ClassificationNets
from .classification.classification_ambiguity_net import ClassificationAmbiguityNet

models_dict={
             'ClassificationNets':ClassificationNets,
             'ClassificationAmbiguityNet':ClassificationAmbiguityNet,
                }

def builder_models(name='VRNetsWithInpainting',**kwargs):
    if name in models_dict.keys():
        return models_dict[name](**kwargs)
    else:
        raise NotImplementedError('{0} not in availables values.'.format(name))