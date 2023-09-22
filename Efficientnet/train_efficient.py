from os import listdir
import cv2
import numpy as np
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D  # Import from TensorFlow
from tensorflow.keras.applications import EfficientNetB0  # Import from TensorFlow
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout  # Import from TensorFlow
from tensorflow.keras.models import Model  # Import from TensorFlow
from tensorflow.keras.callbacks import ModelCheckpoint  # Import from TensorFlow
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import from TensorFlow

raw_folder = "./data/"

def save_data(raw_folder=raw_folder):
    dest_size = (128, 128)
    print("Image processing begin...")

    pixels = []
    labels = []

    # Loop through subfolders in the raw folder
    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print("Folder=", folder)
            # Loop through files in each folder
            for file in listdir(raw_folder + folder):
                if file != '.DS_Store':
                    print("File=", file)
                    pixels.append(cv2.resize(cv2.imread(raw_folder + folder + "/" + file), dsize=(128, 128)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)

    file = open('pix.data', 'wb')
    # Dump information to that file
    pickle.dump((pixels, labels), file)
    # Close the file
    file.close()

    return

def load_data():
    file = open('pix.data', 'rb')

    # Dump information to that file
    (pixels, labels) = pickle.load(file)

    # Close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels

# save_data()
X, y = load_data()
# random.shuffle(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print(X_train.shape)
print(y_train.shape)

def get_model():
    model_effi = EfficientNetB0(include_top=False, weights='imagenet', drop_connect_rate=0.4)
    # Freeze layers
    for layer in model_effi.layers:
        layer.trainable = False

    # Create the model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_effi = model_effi(input)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv')(output_effi)
    x = Flatten(name='flatten')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

effi_model = get_model()

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"

callbacks_list = [
ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
tensorboard_callback
]
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                         rescale=1. / 255,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         brightness_range=[0.2, 1.8], fill_mode="nearest")

aug_val = ImageDataGenerator(rescale=1. / 255)

effi_m = effi_model.fit_generator(aug.flow(X_train, y_train, batch_size=64),
                                 epochs=50,  # steps_per_epoch=len(X_train)//64,
                                 validation_data=aug.flow(X_test, y_test,
                                                          batch_size=64),
                                 callbacks=callbacks_list)

effi_model.save("effib0.h5")


def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    # plt.show()
    plt.savefig('roc.png')
