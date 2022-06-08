import tensorflow as tf
import keras

print(keras.__version__)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
