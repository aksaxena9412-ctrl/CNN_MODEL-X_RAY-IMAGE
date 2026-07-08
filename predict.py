
# this script is for testing the model on a single image
# just to check if it predicts pneumonia or normal correctly

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model/pneumonia_model.h5")

img_path = "test_image.jpeg"   # change this to whatever image you want to test

img_size = 150

img = image.load_img(img_path, target_size=(img_size, img_size))
img_array = image.img_to_array(img)
img_array = img_array / 255.0   # normalizing like we did during training
img_array = np.expand_dims(img_array, axis=0)   # model expects a batch

result = model.predict(img_array)
print(result)

if result[0][0] > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")
