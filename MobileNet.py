import numpy as np                                    # Para abrir o arquivo .npz
import pandas as pd
import tensorflow as tf                               # Para normalizar as imagens
import matplotlib.pyplot as plt
import matplotlib.style as style
import random
import os
import tensorflow_hub as hub

semente = 1264712
np.random.seed(semente)
random.seed(semente)
tf.random.set_seed(semente)
tf.keras.utils.set_random_seed(semente)
os.environ['PYTHONHASHSEED'] = str(semente)
tf.config.experimental.enable_op_determinism()

DATA_PATH = "data/"
TRAIN_IMGS = DATA_PATH + "train/"
VAL_IMGS = DATA_PATH + "val/"
TEST_IMGS = DATA_PATH + "test/"

IMG_SIZE = 128

train_labels = pd.read_csv(DATA_PATH + "labels/train_labels.csv", index_col=False).to_numpy()
val_labels = pd.read_csv(DATA_PATH + "labels/val_labels.csv", index_col=False).to_numpy()
test_labels = pd.read_csv(DATA_PATH + "labels/test_labels.csv", index_col=False).to_numpy()
train_labels = np.ndarray.tolist(train_labels)
val_labels = np.ndarray.tolist(val_labels)
test_labels = np.ndarray.tolist(test_labels)

resize_layer = tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE)
norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_IMGS,
    labels=train_labels,
    color_mode='rgb'
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=VAL_IMGS,
    labels=val_labels,
    color_mode='rgb'
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TEST_IMGS,
    labels=test_labels,
    color_mode='rgb'
)

Data_Augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(factor=(-0.3, 0.3)),
    tf.keras.layers.RandomZoom(
        height_factor=(-0.3, 0.3),
        width_factor=(-0.2, 0.2)
        ),
    tf.keras.layers.RandomTranslation(
        height_factor=(-0.3, 0.3),
        width_factor=(-0.2, 0.2)
        ),
])

train_data = train_data.map(lambda x, y: (Data_Augmentation(x), y))
train_data = train_data.map(lambda x, y: (resize_layer(x), y))
val_data = val_data.map(lambda x, y: (resize_layer(x), y))
test_data = test_data.map(lambda x, y: (resize_layer(x), y))
train_data = train_data.map(lambda x, y: (norm_layer(x), y))
val_data = val_data.map(lambda x, y: (norm_layer(x), y))
test_data = test_data.map(lambda x, y: (norm_layer(x), y))

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost
@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def learning_curves(history):
    """Plot the learning curves of loss and macro f1 score 
    for the training and validation datasets.
    
    Args:
        history: history callback of fitting a tensorflow keras model 
    """
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']
    
    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs+1), loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs+1), macro_f1, label='Training Macro F1-score')
    plt.plot(range(1, epochs+1), val_macro_f1, label='Validation Macro F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('Macro F1-score')
    plt.title('Training and Validation Macro F1-score')
    plt.xlabel('epoch')

    plt.show()
    
    return loss, val_loss, macro_f1, val_macro_f1

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dense(18, activation='sigmoid', name='output')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                loss=macro_soft_f1,
                metrics=[macro_f1, tf.keras.metrics.CategoricalAccuracy()])

history = model.fit(train_data,
             validation_data=val_data,
             batch_size=256,
             epochs=60)

losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history)

model.evaluate(test_data)

model.save("model.h5")