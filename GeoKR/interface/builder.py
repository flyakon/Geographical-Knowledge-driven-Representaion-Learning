'''
@Project : RepresentationLearningLand 
@File    : builder.py
@Author  : Wenyuan Li
@Date    : 2020/11/13 20:34 
@Desc    :  
'''

from .interface_representation import InterfaceRepresentation

interface_dict={
             'InterfaceRepresentation':InterfaceRepresentation,

                }



def builder_models(name='VRNetsWithInpainting',**kwargs):
    if name in interface_dict.keys():
        return interface_dict[name](**kwargs)
    else:
        raise NotImplementedError('{0} not in availables values.'.format(name))
