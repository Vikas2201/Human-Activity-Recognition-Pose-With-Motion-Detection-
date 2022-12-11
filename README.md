# Human-Activity-Recognition-Vision-Based-Pose-With-Motion-Detection
-----------------------------------------------------------------------------------

Modern human activity detection systems are primarily trained and used on video streams and image data. These systems understand variations in data  features and actions that have similar or related behavior. Human activity recognition plays an important role in human-human and human-computer interactions. Manually operated systems are very time consuming and expensive. This project aims to develop a low-cost, high-speed human activity detection system capable of processing both video and images to detect performed activity, which can assist end-users in various applications such as surveillance,  purpose support, etc. I'm here. The system is not only cost-effective, but also  a user-based system that saves time and can be useful for a variety of  activities that require a cognitive process and can be integrated into numerous applications that save a lot of time with precision.

# Proposed System
----------------------------------------------------------

Unlike  existing systems, the proposed system receives input in the form of videos and images and recognizes the activities performed there. This is a much faster and cheaper solution. Detect activity using deep learning. The system can be used, integrated or  further extended to cover a wide range of applications. Therefore, it serves as a base system for various applications and tasks. Additionally, the need for additional personnel for data entry can be reduced. This significantly reduces the company's costs.

# Unique Features of the System
-------------------------------------------------------------

The unique features of the Human Activity Recognition system are listed as follows:

  * The system recognizes activity from both video and image data.

  * It is very easy to use and understand.

  * It is sort of a base system which can be used in various number of applications.

  * It uses cnn based, ConvLSTM and LRCN architecture to solve the solution.

  * It works with a great accuracy on videos dataset to recognize activity in the video. It generates sentence describing the activity on the image dataset with decent accuracy

# Description about dataset
---------------------------------------------------------------

This Action recognition data set of realistic action videos, collected from website, having 101 action categories. With 13320 videos from 101 action categories, UCF101 gives the largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions, etc, it is the most challenging data set to date. As most of the available action recognition data sets are not realistic and are staged by actors, this dataset aims to encourage further research into action recognition by learning and exploring new realistic action categories. The videos in 101 action categories are grouped into 25 groups, where each group can consist of 7 videos of an action. The videos from the same group may share some common features, such as similar background, similar viewpoint, etc.

The movement categories can be divided into five types:
  
  1. human-object interaction

  2. body-only movement

  3. human-human interaction

  4. playing a musical instrument
  
  5. movement
  
Out of these categories we use 'Basketball', 'Biking', 'Bowling', 'PushUps’, 'Typing', 'WalkingWithDog', 'WritingOnBoard' human activity videos datasets for training our model.

# Methodology
------------------------------------------------

![image](https://user-images.githubusercontent.com/76476273/206906282-639bf392-47c4-423b-b4f6-9af1fd6c98c3.png)

  * `Data Acquisition:` It uses web scraping techniques to extract data from websites. Ethical considerations such as privacy and consent issues should be considered when collecting data.

  * `Pre-processing:` Preprocess the dataset. First, read the video file from the dataset, resize the frames of the video to a fixed width and height to reduce computations, and divide the pixel values ​​by 255 to normalize the data to the [0-1] range. Faster convergence when training the network.

       * `Feature Extraction:` The feature extraction step helps create a list containing the resized and normalized frames of the video whose path is passed  as an argument. This function reads the video file frame by frame, but it doesn't add every frame to the list because it only requires an evenly spaced frame sequence length. Feature extraction depends on the type of approach  used for activity detection. This should be done  to reduce the confusion experienced by the model when trained on a large feature set. It also helps  reduce model complexity, improve accuracy and lower error rates. The system extracts important features specific to a particular activity.

       * `Feature Representation:` The extracted features have to be represented in frames (features), class indexes ( labels), and video file paths (video_files_paths) so that the classification algorithms can be easily applied to them.

  * `Splitting the Dataset:` After processing the data, split the dataset. In general, we split the data into two sets: training and validation. You might also create a third holdout dataset called the test set. Otherwise, we often use the terms "verify" and "test" interchangeably. I have the function I need. A NumPy array containing all frames extracted  from the video and another NumPy array one_hot_encoded_labels containing all class labels in  hot encoded form. So we  split the dataset to create a training set and a test set. We  also shuffle the dataset before splitting to avoid  bias and get splits that represent the overall distribution of the data.

  * `Build and Train Model:` Classification algorithms are used to build classification models based on  training data. The model created is  used to test videos for activity detection and classification. Classification algorithms used for activity detection include Naive Bayes, K-Nearest Neighbors, Bayesian Decision Hidden Markov Models (HMM), and Feedforward Neural Networks. Neural networks  and deep learning are also used in recent approaches. Networks such as Convolutional Neural Networks (CNNs), which are used to find  hidden patterns in a given dataset, Recurrent Neural Networks (RNNs), which use time series data to obtain temporal information, and LSTMs is used. Of all available models, we perform human action recognition using LRCN and ConvLSTM approaches.

  * `Evaluation Measures:` Evaluation is measured in terms of Cross Validation and Evaluation Metrics such as Accuracy and Loss of model.

  * `Tuning Hyperparameters:` Performance discrepancies between the training and evaluation sets indicate overfitting and should either reduce the size of the model or add regularization (such as dropout). Poor performance on both training and test sets means poor fit, which may require a larger model or a different learning rate. A common practice is to start with a small model and scale the hyperparameters until you find that training and validation progress differently.

  * `Deployment and analysis:` Once we have trained our model, we may want to use it in the real world. This is especially true when our networks are used by our employees and customers, or when they work behind the scenes with our products and internal tools.
  
# Results
------------------------------------------------

The first approach to training a dataset is the ConvLSTM algorithm. A ConvLSTM cell is a variant of the LSTM network that includes convolutional operations within the network. It is an LSTM with convolution built into its architecture that allows you to identify spatial characteristics of data while considering temporal relationships. After training, we evaluate the model on the test set and evaluate an accuracy of about 0.8365 and a spatial loss of information from the dataset of 1.0015.

![image](https://user-images.githubusercontent.com/76476273/206907786-fbab50fe-b291-448b-9838-ea4272ecf745.png)

  `Total Loss V/S Total Validation Loss`
  
![image](https://user-images.githubusercontent.com/76476273/206907838-47c15a47-8b6c-4613-8807-1d907b213afe.png)

  `Total Accuracy V/S Total Validation Accuracy`
  
![image](https://user-images.githubusercontent.com/76476273/206907878-c3924feb-1455-4c49-a64f-dd10835d374a.png)

A second approach for training datasets is the LRCN algorithm. We implement another approach known as Long-Term Recurrent Convolutional Network (LRCN) that combines CNN and LSTM layers into one model. A convolutional layer is used for spatial feature extraction from frames, and the extracted spatial features are fed to an LSTM layer for temporal sequence modeling at each time step. In this way, the network directly learns spatio-temporal features in end-to-end training and achieves a robust model. After training, we evaluate the model on the test set to assess an accuracy of approximately 0.85785 and a spatial loss of information from the data set of 0.7806.

![image](https://user-images.githubusercontent.com/76476273/206907969-25730f23-5892-487d-b2cc-88f49383cc6b.png)

  `Total Loss V/S Total Validation Loss`

![image](https://user-images.githubusercontent.com/76476273/206908011-490f1421-b6ea-419d-8974-9accea600403.png)

  `Total Accuracy V/S Total Validation Accuracy`
  
![image](https://user-images.githubusercontent.com/76476273/206908037-0dbf8e82-b6a2-4c47-acf8-80d94da1f894.png)

# Conclusion
-----------------------------------------------------------------------------

Computer vision is a trending topic these days, so systems like human activity recognition systems solve a wide range of applications, such as monitoring, monitoring, and assisting the elderly and the blind. very useful and effective for It can be used not only by end users, but also by various organizations to reduce the workload of their employees. This model gives good results for video streams and works well for image data. Activity recognition systems are very important today because of the convenience and problems they provide and solve. Surveillance and surveillance, the need for activity detection such as video segmentation are in high demand where this system can help a lot. The system can be integrated into mobile apps to further assist the elderly and visually impaired. This is an inexpensive, time-saving system that is also prone to human error. This system serves as the base solution for many other activity detection applications. This system is therefore very beneficial for both individuals and organizations for general or special purposes.

# Run Locally
 ------------------------------------------------
 
* Clone the project
```bash
  git clone https://github.com/Vikas2201/Human-Activity-Recognition-Pose-With-Motion-Detection-
```
* Go to the project directory
```bash
  cd Human-Activity-Recognition-Pose-With-Motion-Detection
 ```
* Install dependencies
```bash
  pip install -r requirements.txt
```
* Setting up the Controllers files
```bash
    Update the values inside the Controllers folder
```
* Run the app.py
```bash
  python app.py
```

# Teams Members Name
--------------------------------------------------------------------

 * [@Vikas](https://github.com/Vikas2201)
 * [@Aakash Katyar](https://github.com/XDXD1100189XD)
 * [@Aditya](https://github.com/Aditya-Gahlot)
 * [@Abhijeet 

# Help Me Improve
--------------------------------------------------------------------

Hello Reader if you find any bug please consider raising issue I will address them asap.
