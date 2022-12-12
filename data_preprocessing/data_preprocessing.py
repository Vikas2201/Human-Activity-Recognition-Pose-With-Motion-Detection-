#import required libraries
import cv2
import numpy as np
import logging
import os
import tensorflow as tf
to_categorical = tf.keras.utils.to_categorical

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class Create_DataSet:
    """
        This class is used to fetch data for the training of model.
        Author:
            Vikas
        parameters:
            IMAGE_HEIGHT , IMAGE_WIDTH : Specify the height and width to which each video frame will be resized in our dataset.
            SEQUENCE_LENGTH : Specify the number of frames of a video that will be fed to the model as one sequence.
            DATASET_DIR : Specify the directory containing the dataset.
            CLASSES_LIST : Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.

    """

    def __init__(self, SEQUENCE_LENGTH, IMAGE_HEIGHT , IMAGE_WIDTH, DATASET_DIR, CLASSES_LIST):

        logging.info('Entered in "Create_DataSet" class.')

        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.DATASET_DIR = DATASET_DIR
        self.CLASSES_LIST = CLASSES_LIST

    def frames_extraction(self,video_path):
        '''
            This function will extract the required frames from a video after resizing and normalizing them.
            Args:
                video_path: The path of the video in the disk, whose frames are to be extracted.
            Returns:
                frames_list: A list containing the resized and normalized frames of the video.
        '''

        logging.info('Entered the "frames_extraction" method of the "Create_DataSet" class.')  # logging operation

        try:
            # Declare a list to store video frames.
            frames_list = []

            # Read the Video File using the VideoCapture object.
            video_reader = cv2.VideoCapture(video_path)

            # Get the total number of frames in the video.
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the interval after which frames will be added to the list.
            skip_frames_window = max(int(video_frames_count / self.SEQUENCE_LENGTH), 1)

            # Iterate through the Video Frames.
            for frame_counter in range(self.SEQUENCE_LENGTH):

                # Set the current frame position of the video.
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                # Reading the frame from the video.
                success, frame = video_reader.read()

                # Check if Video frame is not successfully read then break the loop
                if not success:
                    break

                # Resize the Frame to fixed height and width.
                resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
                normalized_frame = resized_frame / 255

                # Append the normalized frame into the frames list
                frames_list.append(normalized_frame)

            # Release the VideoCapture object.
            video_reader.release()

            logging.info('Exited the "frames_extraction" method of the "Create_DataSet" class.')
            # Return the frames list.
            return frames_list

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "frames_extraction" method of the "Create_DataSet" class. Exception message:' + str(
                e))

            logging.info(
                'frames_extraction of videos unsuccessful.Exited the "frames_extraction" method of the "Create_DataSet" class.')

    def create_dataset(self):
        """
            This function will extract the data of the selected classes and create the required dataset.
            Returns:
                features:          A list containing the extracted frames of the videos.
                labels:            A list containing the indexes of the classes associated with the videos.
                video_files_paths: A list containing the paths of the videos in the disk.
        """
        logging.info('Entered the "create_dataset" method of the "Create_DataSet" class.')  # logging operation

        try:
            # Declared Empty Lists to store the features, labels and video file path values.
            features = []
            labels = []
            video_files_paths = []

            # Iterating through all the classes mentioned in the classes list
            for class_index, class_name in enumerate(self.CLASSES_LIST):

                # Display the name of the class whose data is being extracted.
                logging.info(f'Extracting Data of Class: {class_name}')

                # Get the list of video files present in the specific class name directory.
                files_list = os.listdir(os.path.join(self.DATASET_DIR, class_name))

                # Iterate through all the files present in the files list.
                for file_name in files_list:

                    # Get the complete video path.
                    video_file_path = os.path.join(self.DATASET_DIR, class_name, file_name)

                    # Extract the frames of the video file.
                    frames = self.frames_extraction(video_file_path)

                    # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
                    # So ignore the videos having frames less than the SEQUENCE_LENGTH.
                    if len(frames) == self.SEQUENCE_LENGTH:
                        # Append the data to their respective lists.
                        features.append(frames)
                        labels.append(class_index)
                        video_files_paths.append(video_file_path)

            # Converting the list to numpy arrays
            features = np.asarray(features)
            labels = np.array(labels)

            logging.info('Exited the "create_dataset" method of the "Create_DataSet" class.')

            # Return the frames, class index, and video file path.
            return features, labels, video_files_paths

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "create_dataset" method of the "Create_DataSet" class. Exception message:' + str(
                e))
            logging.info(
                'creation of dataframe unsuccessful.Exited the "create_dataset" method of the "Create_DataSet" class.')


    def data_labels(self,labels):
        """
           This function will convert labels into one-hot-encoded vectors.
            Args:
                labels: A list containing the indexes of the classes associated with the videos.
            Returns:
                one_hot_encoded_labels: A Numpy array containing all class labels in one hot encoded format.
        """
        logging.info('Entered the "data_labels" method of the "Create_DataSet" class.')  # logging operation

        try:
            # Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
            one_hot_encoded_labels = to_categorical(labels)

            logging.info('Exited the "data_labels" method of the "Create_DataSet" class.')
            # Return the frames list.
            return one_hot_encoded_labels

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "data_labels" method of the "Create_DataSet" class. Exception message:' + str(
                e))
            logging.info(
                'data label operation unsuccessful.Exited the "data_labels" method of the "Create_DataSet" class.')

