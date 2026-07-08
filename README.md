<<<<<<< HEAD
# Pneumonia Detection using CNN

This is my college project for classifying chest X-ray images as Normal or
Pneumonia using a CNN made with Keras/TensorFlow.

## Dataset
Chest X-Ray Images (Pneumonia) — downloaded from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After downloading, put it in this folder so it looks like:

```
chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

## How to run

1. Install requirements:
```
pip install tensorflow matplotlib numpy
```

2. Train the model:
```
python train_model.py
```

3. Test on a single image:
```
python predict.py
```
(change the `img_path` variable in predict.py to your image)

## What the model does
- Uses 3 convolution + maxpooling layers
- Flatten + Dense layer + Dropout to reduce overfitting
- Final layer is sigmoid since it's just 2 classes (Normal / Pneumonia)
- Trained for 10 epochs

## Results
Accuracy and loss graphs are saved in the `images/` folder after training.
(Fill in your actual test accuracy here after running it)

## What I would improve if I had more time
- Try more epochs / different learning rates
- Try data augmentation more
- Maybe try a bigger dataset for tumor detection too
=======
# CNN_MODEL-X_RAY-IMAGE
>>>>>>> 6d55ff4ad848317e4e43ea9abb014bf65c5a52bb
