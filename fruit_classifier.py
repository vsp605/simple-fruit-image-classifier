import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ---------------------------
# 1. Dataset Paths
# ---------------------------
train_dir = "dataset/train"
test_dir = "dataset/test"

# ---------------------------
# 2. Image Preprocessing
# ---------------------------
IMG_SIZE = (64, 64)  # Smaller size for faster training
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Random rotation
    width_shift_range=0.2,   # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,         # Shear
    zoom_range=0.2,          # Zoom
    horizontal_flip=True,    # Flip images
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ---------------------------
# 3. CNN Model
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# ---------------------------
# 4. Compile Model
# ---------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# 5. Train Model
# ---------------------------
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# ---------------------------
# 6. Plot Training Results
# ---------------------------
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy")
plt.show()

# ---------------------------
# 7. Save Model
# ---------------------------
model.save("fruit_classifier.h5")
print("Model saved as fruit_classifier.h5")
