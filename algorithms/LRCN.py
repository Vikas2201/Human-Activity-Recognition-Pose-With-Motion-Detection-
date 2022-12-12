import logging
import tensorflow as tf
Sequential = tf.keras.models.Sequential
EarlyStopping = tf.keras.callbacks.EarlyStopping
keras = tf.keras.layers

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class lrcn_model:
    """
        This class is used to implement our LRCN architecture.
        Author:
            Vikas
    """

    def __init__(self):
        logging.info('Entered in "LRCN_model" class.')
        pass

    def create_LRCN_model(self,SEQUENCE_LENGTH,IMAGE_HEIGHT,IMAGE_WIDTH,CLASSES_LIST):
        """
            This function will construct the required LRCN model.

            Parameters:
                IMAGE_HEIGHT , IMAGE_WIDTH : Specify the height and width to which each video frame will be resized in our dataset.
                SEQUENCE_LENGTH : Specify the number of frames of a video that will be fed to the model as one sequence.
                CLASSES_LIST : Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.

            Returns:
                model: It is the required constructed LRCN model.
        """
        logging.info('Entered the "create_LRCN_model" method of the "LRCN_model" class.')  # logging operation
        try:
            # We will use a Sequential model for model construction.
            model = Sequential()

            # Define the Model Architecture.
            model.add(keras.TimeDistributed(keras.Conv2D(16, (3, 3), padding='same', activation='relu'),
                                  input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

            model.add(keras.TimeDistributed(keras.MaxPooling2D((4, 4))))
            model.add(keras.TimeDistributed(keras.Dropout(0.25)))

            model.add(keras.TimeDistributed(keras.Conv2D(32, (3, 3), padding='same', activation='relu')))
            model.add(keras.TimeDistributed(keras.MaxPooling2D((4, 4))))
            model.add(keras.TimeDistributed(keras.Dropout(0.25)))

            model.add(keras.TimeDistributed(keras.Conv2D(64, (3, 3), padding='same', activation='relu')))
            model.add(keras.TimeDistributed(keras.MaxPooling2D((2, 2))))
            model.add(keras.TimeDistributed(keras.Dropout(0.25)))

            model.add(keras.TimeDistributed(keras.Conv2D(64, (3, 3), padding='same', activation='relu')))
            model.add(keras.TimeDistributed(keras.MaxPooling2D((2, 2))))
            # model.add(TimeDistributed(Dropout(0.25)))

            model.add(keras.TimeDistributed(keras.Flatten()))

            model.add(keras.LSTM(32))

            model.add(keras.Dense(len(CLASSES_LIST), activation='softmax'))

            # Display the models summary.
            logging.info("Summary of LRCN model : \n" + str(model.summary()))
            # print(model.summary())
            logging.info('Exited the "create_LRCN_model" method of the "LRCN_model" class.')
            # Return the constructed LRCN model.
            return model

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "create_LRCN_model" method of the "LRCN_model" class. Exception message:' + str(
                e))

            logging.info(
                'LRCN Model operation is unsuccessful.Exited the "create_LRCN_model" method of the "LRCN_model" class. method of the "LRCN_model" class.')

    def train_LRCN_model(self,LRCN_model,features_train,labels_train):
        """
            This function will train the LRCN model.
            Parameters:
                convlstm_model : Model of LRCN Architecture
            Returns:
                LRCN_model_training_history : training history of LRCN model.
        """
        logging.info('Entered the "train_LRCN_model" method of the "LRCN_model" class.')  # logging operation

        try:
            # Create an Instance of Early Stopping Callback.
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min',
                                                    restore_best_weights=True)

            # Compile the model and specify loss function, optimizer and metrics to the model.
            LRCN_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

            # Start training the model.
            LRCN_model_training_history = LRCN_model.fit(x=features_train, y=labels_train, epochs=10, batch_size=4,
                                                         shuffle=True, validation_split=0.2,
                                                         callbacks=[early_stopping_callback])
            logging.info('Exited the "train_LRCN_model" method of the "LRCN_model" class.')  # logging operation
            return LRCN_model_training_history

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "train_LRCN_model" method of the "LRCN_model" class. Exception message:' + str(
                e))

            logging.info(
                'train_LRCN_model operation is unsuccessful.Exited the "train_LRCN_model" method of the "LRCN_model" class.')

    def save_LRCN_model(self, LRCN_model):
        """
            This function will help to save the LRCN model.
            Parameters:
                LRCN_model : Model of LRCN
            Returns:
                None thing
        """

        logging.info('Entered the "save_LRCN_model" method of the "LRCN_model" class.')  # logging operation
        try:

            # Define a useful name for our model to make it easy for us while navigating through multiple saved models.
            model_file_name = f'LRCN_model.h5'

            # Save your Model
            LRCN_model.save(model_file_name)
            logging.info('Exited the "save_LRCN_model" method of the "LRCN_model" class.')  # logging operation
            return

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "save_LRCN_model" method of the "LRCN_model" class. Exception message:' + str(
                e))

            logging.info(
                '"save_LRCN_model" operation is unsuccessful.Exited the "save_LRCN_model" method of the "LRCN_model" class.')
