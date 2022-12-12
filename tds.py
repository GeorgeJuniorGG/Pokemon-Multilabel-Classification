import numpy as np                                      # Useful math functions
import tensorflow as tf                                 # Machine learning models and functions
import pandas as pd                                     # Dealing with csv files
import matplotlib.pyplot as plt                         # Plotting
import matplotlib.style as style
import random                                           # Get a seed
import os
from sklearn.model_selection import train_test_split    # Cross validation splitting
import tensorflow_hub as hub                            # Get a model

DATA_PATH = "compiled_data/compiled_pokemon_data.csv"   # Path to the image path/type csv
CLASSES = 18                                            # 18 Pokemon Types
IMG_SIZE = 224                                          # Same size as the model expects the image to be
CHANNELS = 3                                            # RGB images
BATCH_SIZE = 256                                        # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE                # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024                              # Shuffle the training data

# Enable determinism so that we can get the same results
semente = 1264712
np.random.seed(semente)
random.seed(semente)
tf.random.set_seed(semente)
tf.keras.utils.set_random_seed(semente)
os.environ['PYTHONHASHSEED'] = str(semente)
tf.config.experimental.enable_op_determinism()

data = pd.read_csv(DATA_PATH)
X_train, X_temp, y_train, y_temp = train_test_split(data['path'], data['types'], test_size=0.4, random_state=0)    # 60% for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)                     # 20% for validation and 20% for test

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

def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0

    return image_normalized, label

def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    # Create a first dataset of file paths and labels
    labels = labels.to_numpy()
    rotulos = []
    for i in range(len(labels)):
        label = str(labels[i])
        label = label[1:-1]
        label = label.split(',')
        label = [int(x) for x in label]
        rotulos.append(label)
    rotulos = np.asarray(rotulos).astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((filenames, rotulos))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

train_ds = create_dataset(X_train, y_train)

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

feature_extractor_layer.trainable = False           # Freeze the model layers

model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(1024, activation='relu', name='hidden_layer'),
    tf.keras.layers.Dense(CLASSES, activation='sigmoid', name='output')
])

model.summary()

LEARNING_RATE = 1e-5
EPOCHS = 60

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
  loss=macro_soft_f1,
  metrics=[macro_f1])

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val))

losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history)     # Plot F1 loss and F1 score

model.evaluate(create_dataset(X_test, y_test))                              # Evaluate the model in the test dataset

model.save("modelTDS.h5")                                                   # Save the model
