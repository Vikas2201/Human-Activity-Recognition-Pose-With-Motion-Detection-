import os
import cv2
import numpy as np
from collections import deque
import logging

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class prediction:
    """
        This class is used to simply read a video frame by frame from the path passed in as an argument
            and will perform action recognition on video and save the results..
        Author:
            Vikas
    """

    def __init__(self):
        logging.info('Entered in "prediction" class.')
        pass

    def predict_on_video(self,video_file_path, output_file_path, SEQUENCE_LENGTH, model, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST):
        '''
            This function will perform action recognition on a video using the model.
            Args:
            video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
            output_file_path: The path where the output video with the predicted action being performed overlaaded will be stored.
            SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
            model : action recognition model
            IMAGE_HEIGHT , IMAGE_WIDTH : Specify the height and width to which each video frame will be resized in our dataset.
            CLASSES_LIST : list containing the names of the classes used for training
        '''
        logging.info('Entered the "predict_on_video" method of the "prediction" class.')  # logging operation
        try:
            # Initialize the VideoCapture object to read from the video file.
            video_reader = cv2.VideoCapture(video_file_path)

            # Get the width and height of the video.
            original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize the VideoWriter Object to store the output video in the disk.
            video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                       video_reader.get(cv2.CAP_PROP_FPS),
                                       (original_video_width, original_video_height))

            # Declare a queue to store video frames.
            frames_queue = deque(maxlen=SEQUENCE_LENGTH)

            # Initialize a variable to store the predicted action being performed in the video.
            predicted_class_name = ''

            # Iterate until the video is accessed successfully.
            while video_reader.isOpened():

                # Read the frame.
                ok, frame = video_reader.read()

                # Check if frame is not read properly then break the loop.
                if not ok:
                    break

                # Resize the Frame to fixed Dimensions.
                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                normalized_frame = resized_frame / 255

                # Appending the pre-processed frame into the frames list.
                frames_queue.append(normalized_frame)

                # Check if the number of frames in the queue are equal to the fixed sequence length.
                if len(frames_queue) == SEQUENCE_LENGTH:
                    # Pass the normalized frames to the model and get the predicted probabilities.
                    predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]

                    # Get the index of class with highest probability.
                    predicted_label = np.argmax(predicted_labels_probabilities)

                    # Get the class name using the retrieved index.
                    predicted_class_name = CLASSES_LIST[predicted_label]

                # Write predicted class name on top of the frame.
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Write The frame into the disk using the VideoWriter Object.
                video_writer.write(frame)

            logging.info('Exited the "predict_on_video" method of the "prediction" class.')
            # Release the VideoCapture and VideoWriter objects.
            video_reader.release()
            video_writer.release()

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "predict_on_video" method of the "prediction" class. Exception message:' + str(
                e))

            logging.info(
                '"predict_on_video" operation is unsuccessful.Exited the "predict_on_video" method of the "prediction" class. method of the "LRCN_model" class.')

    def predict_single_action(self, video_file_path, SEQUENCE_LENGTH, model, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST):
        '''
            This function will perform single action recognition prediction on a video using the model.
            Args:
                video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
                SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
        '''
        logging.info('Entered the "predict_single_action" method of the "prediction" class.')  # logging operation
        try:
            # Initialize the VideoCapture object to read from the video file.
            video_reader = cv2.VideoCapture(video_file_path)

            # Get the width and height of the video.
            original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Declare a list to store video frames we will extract.
            frames_list = []

            # Initialize a variable to store the predicted action being performed in the video.
            predicted_class_name = ''

            # Get the number of frames in the video.
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate the interval after which frames will be added to the list.
            skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

            # Iterating the number of times equal to the fixed length of sequence.
            for frame_counter in range(SEQUENCE_LENGTH):

                # Set the current frame position of the video.
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                # Read a frame.
                success, frame = video_reader.read()

                # Check if frame is not read properly then break the loop.
                if not success:
                    break

                # Resize the Frame to fixed Dimensions.
                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                normalized_frame = resized_frame / 255

                # Appending the pre-processed frame into the frames list
                frames_list.append(normalized_frame)

            # Passing the  pre-processed frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Display the predicted action along with the prediction confidence.
            print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')


            # Release the VideoCapture object.
            video_reader.release()
            logging.info('Exited the "predict_single_action" method of the "prediction" class.')
            return predicted_class_name

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "predict_single_action" method of the "prediction" class. Exception message:' + str(
                e))

            logging.info(
                '"predict_single_action" operation is unsuccessful.Exited the "predict_single_action" method of the "prediction" class. method of the "LRCN_model" class.')

