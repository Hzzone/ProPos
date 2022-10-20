# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : model_register.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:21 PM 
'''

import logging
import importlib
import os
import pathlib
import re


class Register:

    def __init__(self, registry_name, baseclass=None):
        '''
        model_dict = Register('model_dict', Base)
        @model_dict.register("MTLFace")
        import_models(osp.dirname(osp.abspath(__file__)), 'models')
        Args:
            registry_name:
        '''
        self._dict = {}
        self._name = registry_name
        self._baseclass = baseclass

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if target in self._dict:
            logging.warning(f'Cannot register duplicate ({target})')
        if callable(target):
            # @reg.register
            return add(None, target)

        # @reg.register('alias')
        def class_rebuilder(cls):
            if self._baseclass is not None:
                for p in dir(self._baseclass):
                    if p in dir(cls):
                        continue
                    setattr(cls, p, getattr(self._baseclass, p))
            return add(target, cls)

        return class_rebuilder

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def __repr__(self):
        return str(self._dict)


def import_models(root, prefix):
    '''
    Args:
        root: Path for py
        prefix: import xxx

    Returns:

    '''
    root = os.path.abspath(root)
    for p in pathlib.Path(root).rglob('*.py'):
        p = str(p)
        flag = False
        for x in p.split(os.sep):
            if x.startswith('.'):
                flag = True
        if flag:
            continue
        lib = re.sub(root, prefix, p)
        lib = re.sub(os.sep, '.', lib)[:-3]
        importlib.import_module(lib)
