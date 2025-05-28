import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

val_split = 0.1
val_samples = int(x_train.shape[0] * val_split)
x_val = x_train[:val_samples]
y_val = y_train[:val_samples]
x_train = x_train[val_samples:]
y_train = y_train[val_samples:]

def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()

batch_size = 128
epochs = 15

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val)
)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Тренувальна точність')
    plt.plot(history.history['val_accuracy'], label='Валідаційна точність')
    plt.title('Точність')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Тренувальні втрати')
    plt.plot(history.history['val_loss'], label='Валідаційні втрати')
    plt.title('Втрати')
    plt.xlabel('Епоха')
    plt.ylabel('Втрати')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

score = model.evaluate(x_test, y_test, verbose=0)
print("\nТестові втрати:", score[0])
print("Тестова точність:", score[1])

def plot_predictions(images, labels, predictions, num_images=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        pred_label = np.argmax(predictions[i])
        true_label = np.argmax(labels[i])
        title = f"Pred: {pred_label}\nTrue: {true_label}"
        if pred_label != true_label:
            title = f"❌ {title}"
        else:
            title = f"✅ {title}"
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

sample_images = x_test[:10]
sample_labels = y_test[:10]
predictions = model.predict(sample_images)

plot_predictions(sample_images, sample_labels, predictions)

model.save('mnist_cnn.h5')

# Для завантаження моделі:
# loaded_model = keras.models.load_model('mnist_cnn.h5')