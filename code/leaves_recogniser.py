# Copyright 2022 NXP
#
# NXP Confidential. This software is owned or controlled by NXP and may
# only be used strictly in accordance with the applicable license terms.
# By expressly accepting such terms or by downloading, installing,
# activating and/or otherwise using the software, you are agreeing that
# you have read, and that you agree to comply with and are bound by,
# such license terms.  If you do not agree to be bound by the applicable
# license terms, then you may not retain, install, activate or otherwise
# use the software.
#
 
from deepview.trainer.extensions import interfaces
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.applications import imagenet_utils


def get_plugin():
    return CIFAR10


def preprocess_input(x, **kwargs):
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


class CIFAR10(interfaces.ImageClassificationInterface):
    def get_name(self):
        return "CIFAR10"

    def is_base(self):
        return False

    def get_model(self,
                  input_shape,
                  num_classes,
                  weights=None,
                  named_params={}):
        input_image = layers.Input(shape=input_shape)
        x = input_image

        # x = layers.Conv2D(32, 3, input_shape=[32, 32, 3], padding='same', activation='relu')(x)
        # x = layers.Conv2D(32, 3, activation='relu')(x)
        # x = layers.MaxPooling2D()(x)
        # x = layers.Dropout(0.25)(x)
        # x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        # x = layers.Conv2D(64, 3, activation='relu')(x)
        # x = layers.MaxPooling2D()(x)
        # x = layers.Dropout(0.25)(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(512, activation='relu')(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.Dense(num_classes, activation='softmax')(x)

        x = layers.Conv2D(6, 5, input_shape=[298, 224, 3], activation='relu')(x) # => [294, 220, 6]
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = layers.Conv2D(16, 5, input_shape=[147, 110, 6], activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, ceil_mode=True)(x) # => [71, 53, 16]
        x = layers.Flatten()(x)
        x = layers.Dense(120, activation='relu')(x)
        x = layers.Dense(84, activation='relu')(x)
        # x = layers.Dense(2, activation='softmax')(x)
        x = layers.Dense(2)(x)

        inputs = layer_utils.get_source_inputs(input_image)
        model = models.Model(inputs, x, name='CIFAR10')

        if weights is not None:
            model.load_weights(weights, by_name=True, skip_mismatch=True)

        return model

    def get_task(self):
        return "classification"

    def get_exposed_parameters(self):
        return [
            {
                "name": "Alpha",
                "key": "alpha",
                "default": "0.50",
                "values": ["0.25", "0.50", "0.75", "1.00"],
                "description": "Controls the width of the network"
            },
            {
                "name": "Optimizer",
                "key": "optimizer",
                "default": "Adam",
                "values": ["SGD", "Adam", "RMSprop", "Nadam", "Adadelta", "Adagrad", "Adamax"],
                "description": "Model optimizer"
            }
        ] 

    def get_preprocess_function(self):
        return preprocess_input

    def get_losses(self):
        return ["CategoricalCrossentropy"]

    def get_optimizers(self):
        return ["Adam"]

    def get_metrics(self):
        return ["CategoricalAccuracy"]

    def get_allowed_dimensions(self):
        return ["32", 'Any']

    def get_qat_support(self):
        return [{
            # Per-Channel Quantization
            "supported": "false",
            "types": ['uint8', 'int8', 'float32'],
            "frameworks": ['Tensorflow', 'Converter']
        }, {
            # Per-Tensor Quantization
            "supported": "false",
            "types": ['uint8', 'int8', 'float32'],
            "frameworks": ["Converter"]
        }]

    def get_ptq_support(self):
        return [{
            # Per-Channel Quantization
            "supported": "true",
            "types": ['uint8', 'int8', 'float32'],
            "frameworks": ['Tensorflow', 'Converter']
        }, {
            # Per-Tensor Quantization
            "supported": "true",
            "types": ['uint8', 'int8', 'float32'],
            "frameworks": ["Converter"]
        }]
