import pafy
import logging

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class downloader:
    """
        This class is used to implement our video downloader operation.
        Author:
            Vikas
    """

    def __init__(self):
        logging.info('Entered in "downloader" class.')
        pass

    def download_videos(self,video_url, output_directory):
        '''
            This function downloads the video whose URL is passed to it as an argument.
            Args:
                video_url: URL of the video that is required to be downloaded.
                output_directory:  The directory path to which the video needs to be stored after downloading.
            Returns:
                title: The title of the downloaded video.
       '''
        logging.info('Entered the "download_videos" method of the "downloader" class.')  # logging operation
        try:
            # Create a video object which contains useful information about the video.
            video = pafy.new(video_url)

            # Retrieve the title of the video.
            title = video.title

            # Get the best available quality object for the video.
            video_best = video.getbest()

            # Construct the output file path.
            output_file_path = f'{output_directory}/{title}.mp4'

            # Download the video at the best available quality and store it to the contructed path.
            video_best.download(filepath=output_file_path, quiet=True)

            logging.info('Exited the "download_videos" method of the "downloader" class.')
            # Return the video title.
            return title

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "download_videos" method of the "downloader" class. Exception message:' + str(
                e))

            logging.info(
                '"download_videos" operation is unsuccessful.Exited the "download_videos" method of the "downloader" class. method of the "LRCN_model" class.')
