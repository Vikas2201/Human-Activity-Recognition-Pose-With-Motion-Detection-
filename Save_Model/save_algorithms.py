import pickle
import os
import shutil
import logging
import tensorflow as tf

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class Model_Operation:
    """
        This class shall be used to save the model after training and load the saved model for prediction.
        Author :
            Vikas
    """

    def __init__(self):
        logging.info('Entered in "Model_Operation" class.')
        self.model_directory = 'Model/'

    def save_model(self, model, filename):
        """
            This function help to save the model file into directory system.
            Author:
                Vikas
        """

        logging.info('Entered the "save_model" method of the "Model_Operation" class')  # logging operation
        try:
            path = os.path.join(self.model_directory, filename)
            # create separate directory for each cluster
            if os.path.isdir(path):
                # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path + '/' + filename + '.h5', 'wb') as f:
                model.save(f)
            logging.info('Exited the "save_model" method of the "Model_Operation" class.')  # logging operation
            return "success"

        except Exception as e:
        # logging operation
            logging.error('Exception occurred in "save_model" method of the "Model_Operation" class. Exception message:' + str(
                e))

            logging.info(
                '"save_model" operation is unsuccessful.Exited the "save_model" method of the "Model_Operation" class.')

    def load_model(self, filename):
        """
            This function help to load the model file to memory
            Author :
                Vikas
            Return :
                Model
        """
        logging.info('Entered the "load_model" method of the "Model_Operation" class')  # logging operation
        try:
            logging.info('Model File ' + filename + ' loaded. Exited the "load_model" method of the "Model_Operation" class')
            return tf.keras.models.load_model(filename)

        except Exception as e:
            logging.error(
                'Exception occurred in "load_model" method of the "Model_Operation" class. Exception message:' + str(
                    e))

            logging.info(
                '"load_model" operation is unsuccessful.Exited the "load_model" method of the "Model_Operation" class.')


