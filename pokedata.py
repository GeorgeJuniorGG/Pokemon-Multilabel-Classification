import os                                               # To check if a directory exists
from pathlib import Path                                # To navigate all the folders and subfolders
import re                                               # To select the number from the image names
import matplotlib.pyplot as plt                         # To save the images
import numpy as np                                      # To store the .npz file
import pandas as pd                                     # For reading csv
import tensorflow as tf                                 # For image normalization (size and pixel value)
from sklearn.model_selection import train_test_split    # For cross validation


# Main paths
SPRITES_PATH = 'pokeapi/data/v2/sprites/sprites/pokemon'
TYPES_PATH = 'pokeapi/data/v2/csv/pokemon_types.csv'
TYPES_DICT_PATH = 'pokeapi/data/v2/csv/types.csv'
OUTPUT_PATH = 'compiled_data/'

# Global constants
IMG_SIZE = 128                                                          # Final image size

def normalize_image(imagePath: str):
    """DECPRECATED
    Returns the image resized to IMG_SIZE x IMG_SIZE x 3, with all pixel values in the range [0,1]"""
    img = tf.io.read_file(imagePath)                                    # Read the image
    decoded_img = tf.image.decode_jpeg(img, channels=3)                 # Decode the image to a dense array
    resized_img = tf.image.resize(decoded_img, [IMG_SIZE, IMG_SIZE])    # Redefine the image size
    normalized_img = resized_img/255                                    # Set the pixel values to the range [0,1]
    return normalized_img

def get_type_dict() -> pd.DataFrame:
    """Returns the type dictionary."""
    
    if not os.path.exists('compiled/data/types_dict.csv'):
        types_dict = pd.read_csv(TYPES_DICT_PATH)
        types_dict = types_dict.drop(columns=['generation_id','damage_class_id'])
        types_dict.to_csv(OUTPUT_PATH + 'types_dict.csv', index=False)
    else:
        types_dict = pd.read_csv(OUTPUT_PATH + 'types_dict.csv')
    return types_dict

def get_pokemon_types() -> pd.DataFrame:
    """Returns the pokemon types."""

    if not os.path.exists('compiled/data/pokemon_types.csv'):
        pokemon_types = pd.read_csv(TYPES_PATH)
        pokemon_types = pokemon_types.drop(columns=['slot'])
        pokemon_types.to_csv(OUTPUT_PATH + 'pokemon_types.csv', index=False)
    else:
        pokemon_types = pd.read_csv(OUTPUT_PATH + 'pokemon_types.csv')
    return pokemon_types

def get_one_hot_encoding_type(pokedex_number:int) -> list:
    """Returns the one hot encoding for the pokemon types, given its pokedex number"""

    types = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if pokedex_number == 0:                                                         # Ignore the question mark sprite
        return []

    encoded_types = pokemon_types[pokemon_types['pokemon_id'] == pokedex_number]    # Get all the type ids for the pokemon

    for index,row in encoded_types.iterrows():
        if int(row['type_id']) == 10001 or int(row['type_id']) == 10002:            # Ignore shadow and unknown pokemon types
            return []
        else:
            index = int(row['type_id']) - 1                                         # Get the index to increment the one hot encoding
            types[index] += 1

    return types

def get_image(path:str):
    img = tf.keras.utils.load_img(path, color_mode='rgba')
    img = tf.keras.utils.img_to_array(img)
    
    return img

pokemon_types = get_pokemon_types()
type_dict = get_type_dict()

images = []
labels = []

path = Path(SPRITES_PATH)
for p in path.rglob("*.png"):                                                       # Get all the images in the directory
    try:
        pokedex_number = [int(s) for s in re.findall(r'\b\d+\b', p.name)] [0]       # Get the pokedex number out of the file name
    except:
        continue                                                                    # Ignore sprites that don't have a pokedex number (i.e. egg, substitue, manaphy_egg)

    types = get_one_hot_encoding_type(pokedex_number)
    if types == []:                                                                 # Ignore the shadow and unknown types, as well as the question mark sprite
        continue

    path = str(p.parent) + '/' + p.name
    path = path.replace('\\', '/')                                                  # Get the path for the image

    try:
        img = get_image(path)
        images.append(img)
        labels.append(types)
    except:
        continue

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=0)    # 60% for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)    # 20% for validation and 20% for test

if not os.path.exists('data'):
    os.mkdir('data')
    os.mkdir('data/labels')
    os.mkdir('data/train')
    os.mkdir('data/val')
    os.mkdir('data/test')

train_labels = pd.DataFrame(y_train)
val_labels = pd.DataFrame(y_val)
test_labels = pd.DataFrame(y_test)
train_labels.to_csv("data/labels/train_labels.csv", index=False)
val_labels.to_csv("data/labels/val_labels.csv", index=False)
test_labels.to_csv("data/labels/test_labels.csv", index=False)

for i in range(len(X_train)):
    plt.imsave(f'data/train/{i}.png', X_train[i]/255)
for i in range(len(X_val)):
    plt.imsave(f'data/val/{i}.png', X_val[i]/255)
for i in range(len(X_test)):
    plt.imsave(f'data/test/{i}.png', X_test[i]/255)

np.savez("compiled_data/pokemon_data_cv.npz", X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)
