# Face_Rec

This programm is based on DeepLearning algorithms created with Tensorflow and Keras. Apis like face_recognition are also involved.

It recognizes faces and detects emotions from a video file and writes an output-movie with the recognized faces and emotions frame per frame into the Output-directory.

The face-recognition is realized with the pre-trained face_recognition api, the emotion-detection was created with Tensorflow and Keras. This brings the advantage, that you can take almost any face and the programm will work without changing the code.

In the directories are some example pictures and one video.

If you want to detect other faces you simply have to put a picture of it into the 'Known'-directory and the video you want as output into the 'Unknown'-directoty. Be sure to give your picture the filename you want to appear on the screen.
