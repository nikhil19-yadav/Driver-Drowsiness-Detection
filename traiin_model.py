# train_model.py
# --------------------------------------------
# This script:
# 1. Loads eye images from the "train" folder
# 2. Trains a small CNN to classify OPEN vs CLOSED eyes
# 3. Saves the trained model to model/drowsiness_model.h5
# --------------------------------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path to your dataset folder.
# Inside "train" there should be two subfolders (e.g. Open, Closed)
data_dir = "train"

# ImageDataGenerator does preprocessing + creates batches of images.
# validation_split=0.2 → 80% training, 20% validation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,   # scale pixel values from [0,255] to [0,1]
    validation_split=0.2
)

# Generator that reads images for training
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(24, 24),        # resize all images to 24x24
    color_mode="grayscale",      # use grayscale (1 channel)
    batch_size=32,
    class_mode="binary",         # two classes: open vs closed
    subset="training"            # use 80% data for training
)

# Generator that reads images for validation
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(24, 24),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    subset="validation"          # use 20% data for validation
)

# Define a simple CNN model
model = Sequential([
    # 1st conv layer
    Conv2D(32, (3, 3), activation="relu", input_shape=(24, 24, 1)),
    MaxPooling2D(2, 2),

    # 2nd conv layer
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    # Flatten to a vector
    Flatten(),

    # Fully connected layer
    Dense(128, activation="relu"),
    Dropout(0.5),   # turn off 50% neurons while training to avoid overfitting

    # Output layer: 1 neuron with sigmoid (0 = closed, 1 = open)
    Dense(1, activation="sigmoid")
])

# Compile model: define optimizer, loss function and metrics to track
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Start training the model
# epochs=10 → go through the whole dataset 10 times
model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Save trained model to disk so we can use it later
model.save("model/drowsiness_model.h5")
print("✅ Model saved at model/drowsiness_model.h5")
