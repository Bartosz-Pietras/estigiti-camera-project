from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from utils_tf import *

# Set up the GPU.
use_gpu()

datasets = ["../dataset_equal", "../dataset_original", "../dataset_green", "../dataset_green_equal"]
optimisers = [SGD, Adam]
optimisers_names = ["SGD", "Adam"]
learning_rates = [0.00001, 0.0001]
kernel_sizes = [(3, 3), (5, 5)]


for dataset in datasets:
    for optimiser, optimiser_name in zip(optimisers, optimisers_names):
        for learning_rate in learning_rates:
            for kernel_size in kernel_sizes:
                # Manage directories for training purposes.
                if check_directories(dataset):
                    remove_directories(dataset)
                split_dataset(dataset)

                # Initialise paths to necessary directories.
                paths = {
                    "train": f"{dataset}/train",
                    "valid": f"{dataset}/valid",
                    "test": f"{dataset}/test",
                }

                # Load images, classify them and rescale them.
                classes_list = get_stages_list(get_num_of_stages(dataset))
                train_batches = ImageDataGenerator().flow_from_directory(directory=paths["train"],
                                                                         target_size=(298, 224),
                                                                         classes=classes_list,
                                                                         batch_size=32)

                valid_batches = ImageDataGenerator().flow_from_directory(directory=paths["valid"],
                                                                         target_size=(298, 224),
                                                                         classes=classes_list,
                                                                         batch_size=32)

                test_batches = ImageDataGenerator().flow_from_directory(directory=paths["test"],
                                                                        target_size=(298, 224),
                                                                        classes=classes_list,
                                                                        batch_size=32,
                                                                        shuffle=False)

                # Build the sequential model that will be trained.
                model = Sequential([
                    Conv2D(filters=6, kernel_size=kernel_size, activation='relu', input_shape=(298, 224, 3)),
                    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
                    Conv2D(filters=16, kernel_size=kernel_size, activation='relu'),
                    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
                    Flatten(),
                    Dense(units=len(classes_list), activation='softmax')
                ])

                filepath = f"./model/"
                filepath_weights = f"./weights/"
                model_name = f"model_{dataset[3:]}_{optimiser_name}_lr_{learning_rate}_kernel_size_{kernel_size[0]}"

                checkpoint_model = ModelCheckpoint(filepath=f"{filepath}/{model_name}", monitor='val_accuracy',
                                                   verbose=1, save_best_only=True, mode='max')
                checkpoint_weights = ModelCheckpoint(filepath=f"{filepath_weights}/{model_name}",
                                                     monitor='val_accuracy', verbose=0, save_best_only=True, mode='max',
                                                     save_weights_only=True)

                callbacks = [checkpoint_model, checkpoint_weights]

                # Assign the optimizer, loss function and metrics for the training process.
                model.compile(optimizer=optimiser(learning_rate=learning_rate), loss='categorical_crossentropy',
                              metrics=['accuracy'])

                # Start the training.
                model.fit(x=train_batches, validation_data=valid_batches, epochs=40, verbose=2, callbacks=callbacks)

                # Use the model on test batches.
                predictions = model.predict(x=test_batches, verbose=0)
                np.round(predictions)

                # Create and plot the confusion matrix.
                cm = get_confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
                plot_confusion_matrix(cm, classes_list, model_name)
