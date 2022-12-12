from sklearn.model_selection import train_test_split
import logging

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class Split_DataSet:
    """
        This class is used to split dataset into train and test dataset.
        Author:
            Vikas
        parameters:
            features : A NumPy array containing all the extracted frames of the videos
            one_hot_encoded_labels : A Numpy array containing all class labels in one hot encoded format

    """

    def __init__(self, features, one_hot_encoded_labels):

        logging.info('Entered in "Split_DataSet" class.')
        self.features = features
        self.one_hot_encoded_labels = one_hot_encoded_labels

    def train_test_split(self):
        '''
            This function will split our data to create training and testing sets.
            We will also shuffle the dataset before the split to avoid any bias and get splits representing the overall distribution of the data.

            Returns:
                features_train
                features_test
                labels_train
                labels_test
        '''

        logging.info('Entered the "train_test_split" method of the "Split_DataSet" class.')  # logging operation

        try:
            features_train, features_test, labels_train, labels_test = train_test_split(self.features,
                                                                                        self.one_hot_encoded_labels,
                                                                                        test_size = 0.25, shuffle = True,
                                                                                        random_state = 27)
            logging.info('Exited the "train_test_split" method of the "Split_DataSet" class.')
            # Return the frames list.
            return features_train, features_test, labels_train, labels_test

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "train_test_split" method of the "Split_DataSet" class. Exception message:' + str(
                e))

            logging.info(
                'Operation of spliting the dataset is unsuccessful.Exited the "train_test_split" method of the "Split_DataSet" class.')

