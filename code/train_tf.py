from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from utils_tf import *

# Set up the GPU.
use_gpu()

# Manage directories for training purposes.
if check_directories("../dataset"):
    remove_directories("../dataset")
split_dataset("../dataset")

# Initialise paths to necessary directories.
paths = {
    "train": "../dataset/train",
    "valid": "../dataset/valid",
    "test": "../dataset/test",
}

# Load images, classify them and rescale them.
classes_list = get_stages_list(get_num_of_stages("../dataset"))
train_batches = ImageDataGenerator().flow_from_directory(directory=paths["train"],
                                                         target_size=(298, 224),
                                                         classes=classes_list,
                                                         batch_size=16)

valid_batches = ImageDataGenerator().flow_from_directory(directory=paths["valid"],
                                                         target_size=(298, 224),
                                                         classes=classes_list,
                                                         batch_size=16)

test_batches = ImageDataGenerator().flow_from_directory(directory=paths["test"],
                                                        target_size=(298, 224),
                                                        classes=classes_list,
                                                        batch_size=16,
                                                        shuffle=False)

# Build the sequential model that will be trained.
model = Sequential([
    Conv2D(filters=6, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(298, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=len(classes_list), activation='softmax')
])

model.summary()

# Assign the optimizer, loss function and metrics for the training process.
model.compile(optimizer=SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Start the training.
model.fit(x=train_batches, validation_data=valid_batches, epochs=35, verbose=2)

# Save the model and weights to separate files.
save_model(model=model, name="first_tensorflow_model")
save_weights(model=model, name="first_tensorflow_model_weights_only")

# Use the model on test batches.
predictions = model.predict(x=test_batches, verbose=0)
np.round(predictions)

# Create and plot the confusion matrix.
cm = get_confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
plot_confusion_matrix(cm, classes_list)

