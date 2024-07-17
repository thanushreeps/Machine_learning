# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Set paths
train_dir = r'D:\AI-ML-WIPRO\datasets\train'
val_dir = r'D:\AI-ML-WIPRO\datasets\val'
test_dir = r'D:\AI-ML-WIPRO\datasets\test'

# Visualize
def plot_class_distribution(directory):
    class_names = os.listdir(directory)
    class_counts = [len(os.listdir(os.path.join(directory, class_name))) for class_name in class_names]

    plt.figure(figsize=(8, 6))
    plt.bar(class_names, class_counts, color=['red', 'orange'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Training Set')
    plt.show()

plot_class_distribution(train_dir)

# Data Prepoprocessing
# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values
    shear_range=0.2,           # Randomly shear images
    zoom_range=0.2,            # Randomly zoom in on images
    horizontal_flip=True       # Randomly flip images horizontally
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load,prepo
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),    # Resize images to 150x150
    batch_size=32,             # Number of images to return in each batch
    class_mode='binary'        # Binary classification (normal or pneumonia)
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # First conv layer
    MaxPooling2D(pool_size=(2, 2)),                                    # First max-pooling layer
    Conv2D(64, (3, 3), activation='relu'),                             # Second conv layer
    MaxPooling2D(pool_size=(2, 2)),                                    # Second max-pooling layer
    Conv2D(128, (3, 3), activation='relu'),                            # Third conv layer
    MaxPooling2D(pool_size=(2, 2)),                                    # Third max-pooling layer
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Predictions and Analysis
# #uses trained model to make predictions onthe dtata
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype('int32') #converts the predicted probability to binary classlbls
#if grater than 0.5 it is classified to 1 or else classfied as 0.

# Classification ,lables
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Visualize Training History
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.title('Training and Validation Accuracy/Loss')
plt.savefig('training_history.png')
plt.show()