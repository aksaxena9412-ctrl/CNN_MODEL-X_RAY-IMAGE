
# Pneumonia Detection using CNN
# I made this for my college project
# dataset used: Chest X-Ray Images (Pneumonia) from kaggle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# path to dataset (i downloaded it and put it in a folder called chest_xray)
train_path = "chest_xray/train"
val_path = "chest_xray/val"
test_path = "chest_xray/test"

img_size = 150   # resizing all images to 150x150
batch_size = 32

# using ImageDataGenerator to load images and also do some augmentation
# so the model doesnt overfit
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

training_set = train_gen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

val_set = test_gen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

test_set = test_gen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

print(training_set.class_indices)   # just to check which class is 0 and which is 1


# building the CNN model
# i used 3 conv layers, that seemed to give ok results after trying a few things
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))   # adding dropout so it doesnt overfit too much
model.add(Dense(1, activation='sigmoid'))   # sigmoid because its just 2 classes

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# training the model
epochs = 10   # tried 20 first but it was taking too long so reduced it

history = model.fit(
    training_set,
    validation_data=val_set,
    epochs=epochs
)

# checking how well it does on test data
test_loss, test_accuracy = model.evaluate(test_set)
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# saving the model so i dont have to train again everytime
model.save("model/pneumonia_model.h5")
print("model saved!")

# plotting accuracy and loss graphs
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/accuracy_plot.png')
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/loss_plot.png')
plt.show()
