import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths - Update these paths as needed
BASE_DIR = r"C:\Users\abhin\Downloads\archive (17)\xray_dataset_covid19"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
CUSTOM_IMAGE_PATH = r"C:\Users\abhin\Downloads\download.jpg"  # Your custom image

# Image parameters
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25

# -------------------- Data Preprocessing --------------------
def prepare_data():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    training_set = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_set = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return training_set, test_set

# -------------------- CNN Model --------------------
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------- Prediction on Custom Image --------------------
def predict_image(model, image_path, class_indices):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    labels = dict((v, k) for k, v in class_indices.items())
    
    predicted_label = labels[int(round(prediction))]
    print(f"Prediction: {predicted_label.upper()}")
    return predicted_label

# -------------------- Main Flow --------------------
if __name__ == "__main__":
    training_set, test_set = prepare_data()
    model = build_model()
    model.fit(training_set, validation_data=test_set, epochs=EPOCHS)

    print("\n--- Predicting Custom Image ---")
    predict_image(model, CUSTOM_IMAGE_PATH, training_set.class_indices)