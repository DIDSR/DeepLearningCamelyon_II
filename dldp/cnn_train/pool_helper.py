# This code is adopted from https://gist.githubusercontent.com/joelouismarino/a2ede9ab3928f999575423b9887abd14/raw/0054c12aeb27b86225bc793fc19366665867a0cf/pool_helper.py
import os
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.layers.core import Layer


class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
