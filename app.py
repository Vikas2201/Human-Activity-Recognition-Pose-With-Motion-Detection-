import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from Save_Model.save_algorithms import Model_Operation
from training import Training
from Prediction.prediction_on_video import prediction
from Video_Downloader.video_downloader import downloader

UPLOAD_FOLDER = 'static/test_videos/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Inital Variables
IMAGE_HEIGHT, IMAGE_WIDTH = 81, 81
SEQUENCE_LENGTH = 50
DATASET_DIR = "dataset/UCF-101"
CLASSES_LIST = ['Basketball', 'Biking', 'Bowling', 'PushUps', 'Typing', 'WalkingWithDog', 'WritingOnBoard']

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        url = secure_filename(file.filename)

        # Download a Video.
        video_title = downloader.download_videos(url, UPLOAD_FOLDER)

        # Get the Video's path we just downloaded.
        input_video_file_path = f'{UPLOAD_FOLDER}/{video_title}.mp4'

        # start training model
        Train_model = Training()
        FileName = Train_model.train(IMAGE_HEIGHT, IMAGE_WIDTH,SEQUENCE_LENGTH,DATASET_DIR,CLASSES_LIST)

        # load the model for testing
        MO = Model_Operation()
        model = MO.load_model(FileName)

        predict = prediction()
        # prediction
        first_name = request.form.get("options")
        output , filename = None , None
        if str(first_name) == 'Multiple action':
            print('Multiple')
            # Construct the output video path.
            output_video_file_path = f'{UPLOAD_FOLDER}/{video_title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
            # Perform Action Recognition on the Test Video.
            predict.predict_on_video(input_video_file_path,output_video_file_path, SEQUENCE_LENGTH, model, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)
            filename = output_video_file_path
        elif str(first_name) == 'Single Action':
            print('Single')
            # Perform Single Action Recognition on the Test Video.
            output = predict.predict_on_video(input_video_file_path, SEQUENCE_LENGTH, model, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)
            filename = input_video_file_path

        print("Prediction : ", output)
        print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        return render_template('upload.html', filename= filename)

@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run()
