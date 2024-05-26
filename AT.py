import cv2
import numpy as np
import os
from keras.models import load_model
import time
from pygame import mixer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Layer, Add, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
from tensorflow.keras import initializers

# Initialize pygame mixer for alarm
mixer.init()
sound = mixer.Sound('/Users/subhamgupta/Downloads/Drowsiness detection/alarm.wav')

# Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the Channel and Spatial Attention Layers
class ChannelAttention(Layer):
    def __init__(self, ratio=8):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_dense_one = Dense(input_shape[-1] // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.shared_dense_two = Dense(input_shape[-1], kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, inputs.shape[-1]))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = Reshape((1, 1, inputs.shape[-1]))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        return Add()([avg_pool, max_pool])

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv2d = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return self.conv2d(concat)

class CBAMBlock(Layer):
    def __init__(self, ratio=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = Multiply()([inputs, x])
        x = self.spatial_attention(x)
        return Multiply()([inputs, x])

# Define the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
        CBAMBlock(),
        Conv2D(64, (3, 3), activation='relu'),
        CBAMBlock(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights or train your model
# model.load_weights('path_to_model_weights.h5')

# Video capture from webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (24, 24))
            eye = eye / 255.0
            eye = eye.reshape(24, 24, 1)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)
            if np.argmax(prediction) == 1:
                cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                score = score - 1
            else:
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                score = score + 1

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        # Trigger the alarm
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            sound.play()
        except:  # Handle sound play exception if needed
            pass
        if thicc < 16:
            thicc += 2
        else:
            thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Load the model
    model = load_model('/Users/subhamgupta/Downloads/Drowsiness detection/models/cnnCat2.h5')
    model.summary()# Adjust path as needed

    # Specify the path to your image
    image_path = '/Users/subhamgupta/Downloads/Drowsiness detection/models/image.jpg'  # Adjust the path to your image

    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Reads image as grayscale directly

    if image is None:
        raise ValueError("Could not open or find the image")

    # Resize the image to the size required by your model, e.g., 24x24 for the eye detection model
    image = cv2.resize(image, (24, 24))

    # Normalize the pixel values to [0,1]
    image = image / 255.0

    # Expand dimensions to match the model's input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Adds a batch dimension
    image = np.expand_dims(image, axis=-1)  # Adds a channel dimension if your model expects it

    # Measure the time it takes to make predictions
    start_time = time.time()
    predictions = model.predict(image)
    elapsed_time = time.time() - start_time

    print(f"Time taken for prediction: {elapsed_time:.3f} seconds")
    print(f"Predictions: {predictions}")


cap.release()
cv2.destroyAllWindows()

#%%
