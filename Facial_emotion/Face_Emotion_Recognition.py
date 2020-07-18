import face_recognition
import cv2
import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model from emotion recognition
model = model_from_json(open("fer.json", "r").read())
#load weights from emotion recognition
model.load_weights('fer.h5')


output_dir= 'Output'
unknown_dir= 'Unknown'

# function to resize (this case only the input_picture from the known_dir)
def read_img(path):
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    width = video_width
    ratio = width / float(w)
    height = int(h*ratio)

    return cv2.resize(img, (width, height))


# Opens the input movie from the unknown_dir
for file in os.listdir(unknown_dir):
    # read input_movie 
    input_movie = cv2.VideoCapture(unknown_dir + '/' + file)
    
    # Gets dimensions of the input-movie (framerate/resolution)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    fps= input_movie.get(cv2.CAP_PROP_FPS)
    video_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')                
    # renames the output file if it already exists          
    num= 0                                                  
    output_filename= 'video_out(' + str(num) + ').avi'      
    for output_filename in os.listdir(output_dir):          
        int(num)                                            
        num += 1                                            
        print(output_filename)                              
        print(num) 
        
    # Creates an output movie file                                         
    output_movie = cv2.VideoWriter('Output/video_out(' + str(num) + ').avi', fourcc, fps, (video_width, video_height))

    
# Lists and Variables
known_dir='Known'
known_faces = []
face_name = []
    
# loads the pictures from the known_dir and does the faceencodings
# appends it to the defined lists
for file in os.listdir(known_dir):
    image_read = read_img(known_dir + '/' + file)
    face_encoding = face_recognition.face_encodings(image_read)[0]
    known_faces.append(face_encoding)
    face_name.append(file.split('.')[0]) # Get all file-names and save them 
    # be sure to give your files the names from the people you want to detect

# Lists and Variables
face_locations = []
face_encodings = []
frame_number = 0

while True:
    # Grabs a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quits when the input video file ends
    if not ret:
        break

    # Converts the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Finds all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # List
    face_names = []
    
    for face_encoding in face_encodings:
        # Compares all faces and saves if it's a match!
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # if there is no match the name should be none
        name = None
        
        # if there is a match this tells you the name of the matched person/ file name
        for i in range(len(match)):
            if match[i]:
                name = face_name[i]
                face_names.append(name)
    
            print(face_names)   # prints it's solution for every frame in the console
    # Labels the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draws a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Draws a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 20, top - 2), font, 0.5, (255, 0, 0), 1)
  
    # Converts the input-frame from BGR to GRAY
    gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # all possible emotions to be recognized
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = 0
    

    for (top,right,bottom,left) in face_locations:
        # prints recognized emotion in to the console for every frame
        if not emotions:
            continue
        # resize the input-frame to 48x48 like the photos from the training
        roi_gray=gray_img[right:right+bottom,top:top+left]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
            
        predictions = model.predict(img_pixels)
    
        #find max indexed array
        max_index = np.argmax(predictions[0])
    
        predicted_emotion = emotions[max_index]
        print(predicted_emotion)
        
        
        # Draws a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Draws a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 255), cv2.FILLED)
        #put emotion as Text into the frame
        cv2.putText(frame, predicted_emotion, (int(left), int(bottom)),cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0,0,255), 2)


    # Writes the results frame per frame into the output file and prints each number of it in the console
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# open and close window
input_movie.release()
cv2.destroyAllWindows()