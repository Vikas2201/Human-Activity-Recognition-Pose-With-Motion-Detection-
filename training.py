import os
from data_preprocessing.data_preprocessing import Create_DataSet
from Split_dataset.train_test_split import Split_DataSet
from algorithms.ConvLSTM import Convlstm_model
from algorithms.LRCN import lrcn_model
from Evaluation.evaluation import Metrics
from Save_Model.save_algorithms import Model_Operation
import logging

# configuring logging operations
logging.basicConfig(filename='deployment_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class Training:
    """
        This class is used to perfrom training operations.
        Author:
            Vikas
    """
    def __init__(self):
        logging.info('Entered in "Training" class.')
        pass

    def train(self, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, DATASET_DIR, CLASSES_LIST):
        '''
            This function will perform training of dataset.
            return :
                filename of save model.
        '''
        logging.info('Entered the "train" method of the "Training" class.')  # logging operation
        try:
            dataset = Create_DataSet(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, DATASET_DIR, CLASSES_LIST)
            features, labels, video_files_paths = dataset.create_dataset()
            one_hot_encoded_labels = dataset.data_labels(labels)
            SD = Split_DataSet(features,one_hot_encoded_labels)
            features_train, features_test, labels_train, labels_test = SD.train_test_split()
            model1 = Convlstm_model()
            # Construct the required convlstm model.
            convlstm_model = model1.create_convlstm_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)
            convlstm_model_training_history = model1.train_ConvLSTM_model(convlstm_model,features_train,labels_train)

            model2 = lrcn_model()
            LRCN_model = model2.create_LRCN_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)
            LRCN_model_training_history = model2.train_LRCN_model(LRCN_model,features_train,labels_train)

            metric = Metrics()
            loss1, accuracy1 = metric.accuracy_and_loss(features_test, labels_test, convlstm_model)
            loss2, accuracy2 = metric.accuracy_and_loss(features_test, labels_test, LRCN_model)
            mo = Model_Operation()
            if loss1 <= loss2 and accuracy1 >= accuracy2 :
                mo.save_model(convlstm_model,"Convlstm_Model")
                filename = "Convlstm_Model.h5"
            else:
                mo.save_model(LRCN_model,"LRCN_Model")
                filename = "LRCN_Model.h5"

            #model2.save_LRCN_model(LRCN_model)
            #filename = "LRCN_Model.h5"

            logging.info('Exited the "train" method of the "Training" class.')
            return filename

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "train" method of the "Training" class. Exception message:' + str(
                e))

            logging.info(
                '"train" operation is unsuccessful.Exited the "train" method of the "Training" class.')

