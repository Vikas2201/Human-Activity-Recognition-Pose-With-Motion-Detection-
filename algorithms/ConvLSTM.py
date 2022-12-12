import logging
import tensorflow as tf
Sequential = tf.keras.models.Sequential
EarlyStopping = tf.keras.callbacks.EarlyStopping
keras = tf.keras.layers

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class Convlstm_model:
    """
        This class is used to implement Keras ConvLSTM2D recurrent layers.
        Author:
            Vikas
    """

    def __init__(self):
        logging.info('Entered in "convlstm" class.')
        pass

    def create_convlstm_model(self,SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,CLASSES_LIST):
        """
            This function will construct the required convlstm model.
            Parameters:
                IMAGE_HEIGHT , IMAGE_WIDTH : Specify the height and width to which each video frame will be resized in our dataset.
                SEQUENCE_LENGTH : Specify the number of frames of a video that will be fed to the model as one sequence.
                CLASSES_LIST : Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
            Returns:
                model: It is the required constructed convlstm model.
        """
        logging.info('Entered the "create_convlstm_model" method of the "convlstm_model" class.')  # logging operation
        try:

            # We will use a Sequential model for model construction
            model = Sequential()

            # Define the Model Architecture.

            model.add(keras.ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                             recurrent_dropout=0.2, return_sequences=True, input_shape=(SEQUENCE_LENGTH,
                                                                                        IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

            model.add(keras.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
            model.add(keras.TimeDistributed(keras.Dropout(0.2)))

            model.add(keras.ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                             recurrent_dropout=0.2, return_sequences=True))

            model.add(keras.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
            model.add(keras.TimeDistributed(keras.Dropout(0.2)))

            model.add(keras.ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                             recurrent_dropout=0.2, return_sequences=True))

            model.add(keras.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
            model.add(keras.TimeDistributed(keras.Dropout(0.2)))

            model.add(keras.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                             recurrent_dropout=0.2, return_sequences=True))

            model.add(keras.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

            model.add(keras.Flatten())

            model.add(keras.Dense(len(CLASSES_LIST), activation="softmax"))

            # Display the models summary.
            logging.info("Summary of model : \n" + str(model.summary()))
            #print(model.summary())
            logging.info('Exited the "create_convlstm_model" method of the "convlstm_model" class.')
            # Return the constructed convlstm model.
            return model

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "create_convlstm_model" method of the "convlstm_model" class. Exception message:' + str(
                e))

            logging.info(
                'ConvLSTM Model operation is unsuccessful.Exited the "create_convlstm_model" method of the "convlstm_model" class.')

    def train_ConvLSTM_model(self,convlstm_model,features_train,labels_train):
        """
            This function will train the ConvLSTM model.
            Parameters:
                convlstm_model : Model of ConvLSTM
            Returns:
                convlstm_model_training_history : training history of ConvLSTM model.
        """
        logging.info('Entered the "train_ConvLSTM_model" method of the "convlstm_model" class.')  # logging operation

        self.labels_train = labels_train
        self.features_train = features_train
        try:
            # Create an Instance of Early Stopping Callback
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                                                    restore_best_weights=True)

            # Compile the model and specify loss function, optimizer and metrics values to the model
            convlstm_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

            # Start training the model.
            convlstm_model_training_history = convlstm_model.fit(x=self.features_train, y=self.labels_train, epochs=70,
                                                                 batch_size=4,
                                                                 shuffle=True, validation_split=0.2,
                                                                 callbacks=[early_stopping_callback])
            logging.info('Exited the "train_ConvLSTM_model" method of the "convlstm_model" class.')  # logging operation
            return convlstm_model_training_history

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "train_ConvLSTM_model" method of the "convlstm_model" class. Exception message:' + str(
                e))

            logging.info(
                'train_ConvLSTM_model operation is unsuccessful.Exited the "train_ConvLSTM_model" method of the "convlstm_model" class.')

    def save_convlstm_model(self,convlstm_model):
        """
            This function will train the ConvLSTM model.
            Parameters:
                convlstm_model : Model of ConvLSTM
            Returns:
                None thing
        """
        logging.info('Entered the "save_convlstm_model" method of the "convlstm_model" class.')  # logging operation
        try:

            # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
            model_file_name = f'convlstm_model.h5'

            # Save your Model
            convlstm_model.save(model_file_name)
            logging.info('Exited the "save_convlstm_model" method of the "convlstm_model" class.')  # logging operation
            return

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "save_convlstm_model" method of the "convlstm_model" class. Exception message:' + str(
                e))

            logging.info(
                '"save_convlstm_model" operation is unsuccessful.Exited the "save_convlstm_model" method of the "convlstm_model" class.')
