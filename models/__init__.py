import importlib
import os
import os.path as osp
from utils.model_register import import_models, Register

model_dict = Register('model_dict')

import_models(osp.dirname(__file__), 'models')
import_models(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'experiment'), 'experiment')
