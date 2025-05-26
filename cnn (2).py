import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Load the pixelated image
image_path = r"E:\airplane.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x.astype('float32') / 255

# Convert from BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image using bicubic interpolation
height, width = image.shape[:2]
scale_factor = 10  # Scale up the image by a factor
new_width = width * scale_factor
new_height = height * scale_factor

# Reconstruct the image
reconstructed_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Apply a Gaussian blur to smooth out pixelation
smoothed_image = cv2.GaussianBlur(reconstructed_image, (15, 15), 0)

# Combine the smoothed image and reconstructed image for enhancement
blended_image = cv2.addWeighted(reconstructed_image, 0.6, smoothed_image, 0.4, 0)

# Display the original, smoothed, and blended images
plt.figure(figsize=(18, 6))

# Original image
plt.subplot(1, 3, 1)
plt.title("Original Pixelated Image")
plt.imshow(image_rgb)
plt.axis('off')

# Reconstructed image
plt.subplot(1, 3, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image)
plt.axis('off')

# Blended image
plt.subplot(1, 3, 3)
plt.title("Blended Image (Enhanced)")
plt.imshow(blended_image)
plt.axis('off')

# Prediction using the model
preds = model.predict(x)
pred_class = np.argmax(preds, axis=1)

# Plot the image with the predicted class name
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.title(class_names[pred_class[0]])
plt.show()
