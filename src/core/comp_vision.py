# todo: 'Satellite images have large sizes. You should think about how to process them in
#   order not to lose the quality.' --> think on how to avoid overfitting

import os

import cv2

from PIL import Image

from loguru import logger

import numpy as np
import pandas as pd
from pandas import DataFrame

import geopandas as gpd

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize

from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

from keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from core.consts import GEO_DATASET, IMAGE_PATH


class ImageIdentification:

    async def dataset_prep(self):
        """
        Prepares dataset by reading GeoJSON and creating masks.
        Here we use rasterio python library to read the file.
        """
        with rasterio.open(IMAGE_PATH, 'r', driver='JP2OpenJPEG') as img:
            raster_img = reshape_as_image(img.read())
            raster_meta = img.meta()
            logger.info(raster_meta)

            plt.figure(figsize=(15,15))
            plt.imshow(raster_img)

    def create_mask(self, image, polygons):
        """ Create a binary mask from polygons on the image """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for polygon in polygons:
            shapes = [(mapping(polygon), 1)]  # Use 1 for deforestation areas
            mask = rasterio.features.rasterize(shapes, out_shape=image.shape[:2], fill=0)
        return mask

    async def build_cnn(self, input_shape=(224, 224, 3)):
        """
        For objects detection on pictures, we apply CNN deep learning algorithm,
        since it is one of the best to use for image (!) analysis.

        """
        model = Sequential()

        # First convolutional layer
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second convolutional layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third convolutional layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten and fully connected layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))  # Dropout for regularization

        # Output layer (binary classification for deforestation or not)
        model.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    async def run(self):
        """ Main method to load images, create masks, and train the model. """
        # Step 1: Preprocess dataset
        df = await self.dataset_prep()  # Load GeoJSON data
        logger.info(df)
        image_path = IMAGE_PATH

        # Step 2: Load and process the satellite image
        with rasterio.open(image_path) as src:
            # Read the RGB bands (Red, Green, Blue)
            image = reshape_as_image(src.read([1, 2, 3]))  # Read 3 bands: Red, Green, Blue
            image = cv2.resize(image, (224, 224))  # Resize image to 224x224 for CNN input

            # Create a binary mask based on the polygons in GeoJSON
            mask = self.create_mask(image, df['geometry'])

        # Step 3: Prepare training data
        images = [image]  # List of satellite images (224x224x3)
        labels = [mask]   # Corresponding labels (binary masks)

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Convert images and masks to NumPy arrays
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        # Step 4: Build and train the CNN model
        model = await self.build_cnn()  # Build the CNN model

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        # Step 5: Save the trained model
        model.save('deforestation_cnn_model.h5')
        logger.info('Training complete and model saved!')

