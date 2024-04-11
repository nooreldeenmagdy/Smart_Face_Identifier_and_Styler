import pickle
import os
import sys
from skimage.filters import threshold_otsu



# for style transfer
from imutils.video import VideoStream

import itertools
import imutils
import time
import mediapipe as mp


#import mysql.connector, os

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.core.text import LabelBase


import cv2 as cv
import face_recognition
import numpy as np
from utils import image_resize
from identifier import get_faces_data
from collections import deque






# Declare dependencies
#Paths
sys.path.append('.')
kivy.resources.resource_add_path('.')

#SQL Database
##connection = mysql.connector.connect(
##  
##    host="localhost",
##    user="user",
##    password="rootroot9",
##    database = "smart_camera",
##    port=3306)
##
##cursor = connection.cursor()

# Kivy Font
LabelBase.register(name = 'OpenSans', fn_regular = 'OpenSans-Regular.ttf')

# Cascades
face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
eyes_cascade = cv.CascadeClassifier('cascades/frontalEyes35x16.xml')
nose_cascade = cv.CascadeClassifier('cascades/Nose18x15.xml')

# Faces data
with open('faces-data.pickle', 'rb') as file:
    faces_data = pickle.load(file)

# Glasses + Mustache files
#glasses = cv.imread("filters/glasses.png", -1)
#mustache = cv.imread('filters/mustache.png',-1)

# Painter globals
# Define the upper and lower boundaries for a color to be considered "Blue"
blue_lower = np.array([100, 60, 60])
blue_upper = np.array([140, 255, 255])
# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)
# Setup deques to store separate colors in separate arrays
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
bindex = 0
gindex = 0
rindex = 0
# Drawing setting
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0), (120,120,120)]
colorIndex = 0
font = cv.FONT_HERSHEY_SIMPLEX

# Creates DB Tables

##def db_creation():
##
##	gallary = "CREATE TABLE IF NOT EXISTS capture (\
##        pic_id int not null primary key AUTO_INCREMENT,\
##        photo longblob not null,\
##        time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
##	face_ids = "CREATE TABLE IF NOT EXISTS face_ids( \
##        name varchar(200) not null,face_id int not null primary key AUTO_INCREMENT,face_enc longblob not null, \
##        time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
##
##
##	faces = "CREATE TABLE IF NOT EXISTS face_id(\
##        name varchar(200) not null,\
##        pic_id int primary key,\
##        foreign key (pic_id) references capture(pic_id))"
##
##	cursor.execute(gallary)
##	connection.commit()
##	cursor.execute(face_ids)
##	connection.commit()
##
##	cursor.execute(faces)
##	connection.commit()
##
##	
##
##db_creation()





class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv.VideoCapture(0)
        #self.capture = VideoStream(src=0).start()
        self.triger(self.update_cam)
        self.not_testing = True
        self.have_face = False
        self.saved_face = False
        self.styling = True



    

    def triger(self, triger_mode):
        Clock.unschedule(self.update_cam)
        Clock.unschedule(self.detect_faces)
        Clock.unschedule(self.identify_faces)
        Clock.unschedule(self.glasses)
        Clock.unschedule(self.painter)
        Clock.unschedule(self.style_transfer)
        Clock.schedule_interval(triger_mode, 1/60)

        
    def update_cam(self, dt):
        self.styling = True
        ret = False
    	#capture.release()
        ret, frame = self.capture.read()
        #print(ret)
        if self.not_testing:
                

            frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)

            #convert frame to texture
            buf = cv.flip(frame, 0)
            #buf = buf.tostring()
            buf = buf.tobytes()

            texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # display frame from the texture
            self.ids.cam.texture = texture_f 

	      # testing   
        return ret


    
    
    
    
    # Detection: Detect the human face

    def detect_faces(self, dt):
        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        
        
       
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray,
                                             # scaleFactor=1.1,
                                             # minNeighbors=10,
                                             # minSize=(100, 100),
                                             # flags=cv.CASCADE_SCALE_IMAGE)

        # for (x,y,w,h) in faces:
            # cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        
        
        def draw_landmarks(image, results):
            # Pose
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        def mediapipe_detection(frame, model):
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # works on RGB not BGR
            frame.flags.writeable = False # To improve performance can work without it
            results = model.process(frame) 
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR) # Back to BGR
            return frame, results

        
        with mp_holistic.Holistic() as holistic:

            h, w, _ = frame.shape

            frame, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_landmarks(frame, results)
            
            # Draw Rectangle around the face
            padding = 30
            start_point = (int(results.pose_landmarks.landmark[6].x * w)-padding, int(results.pose_landmarks.landmark[6].y * h)-padding)
            end_point = (int(results.pose_landmarks.landmark[9].x * w)+padding, int(results.pose_landmarks.landmark[9].y * h)+padding)
            # We multiply by w and h to use the real x and y of the frame
            
            if(results.pose_landmarks.landmark[0]):
                cv.rectangle(frame, start_point, end_point, (255, 0, 0) , 2)


        buf = cv.flip(frame, 0)
        buf = buf.tobytes()
        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.cam.texture = texture_f


 		# testing
        #if len(faces) != 0:
        	
        	#self.have_face = True

    # Identification: Identify the human name

    
    def identify_faces(self, dt):
        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=10,
                                             minSize=(100, 100),
                                             flags=cv.CASCADE_SCALE_IMAGE)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []

        # testing
        if len(faces) != 0:
        	self.have_face = True

        for encoding in encodings:
            matches = face_recognition.compare_faces(faces_data["encodings"], encoding)
            name = "Unknown"

            if True in matches:

                #Find positions at which we get True and store them
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
				# testing
                self.saved_face = True


                # loop over the matched indexes and maintain a count for each recognized face face
                for i in matched_idxs:
                    #Check the names at respective indexes we stored in matched_idxs
                    name = faces_data["names"][i]
                    #increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                #set name which has highest count
                name = max(counts, key=counts.get)
 
            # update the list of names
            names.append(name)
            # loop over the recognized faces
            for (x, y, w, h), name in zip(faces, names):
                # draw the predicted face name on the image
                cv.rectangle(frame, (x, y),(x+w, y+h),(0,255,0),2)
                cv.rectangle(frame, (x-10, y+h),(x+w+10, y+int(h*1.15)),(0,255,0), -1)
                cv.putText(frame, name, (x-5, y+int(h*1.11)), cv.FONT_HERSHEY_SIMPLEX, w/250, (255, 255, 255), 2)

        buf = cv.flip(frame, 0)
        buf = buf.tobytes()
        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.cam.texture = texture_f 

    
    # Capturing and saving photos.

    
    def capture_screen(self):
            
    	# store as file in folder
        ret, frame = self.capture.read()
        DIR = './gallery'
        images = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        count = len(images)
        while 'capture{}.jpg'.format(count) in images:
                count+=1
        cv.imwrite('gallery/capture{}.jpg'.format(count), frame)

        
##        # store as blob in DB
##        with open('gallery/capture{}.jpg'.format(count), "rb") as file:
##                bData = file.read()
##                
##        sqlStatment = "INSERT INTO capture (photo) VALUES (%s)"
##        cursor.execute(sqlStatment, (bData, ))
##        connection.commit()

    # Capturing and saving videos.
    
    def capture_video(self):

        # Create an object to read 
        # from camera
        
        # We need to set resolutions.
        # so, convert them from float to integer.
        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
   
           
        size = (frame_width, frame_height)
           
        # Below VideoWriter object will create
        # a frame of above defined The output 
        # is stored in 'filename.avi' file.

        DIR = './gallery'
        images = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        count = len(images)
        while 'capture{}.avi'.format(count) in images:
                count+=1



        result = cv.VideoWriter('gallery/capture{}.avi'.format(count), 
                                 cv.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
            
        while True:
                
            ret, frame = self.capture.read()
          
            if ret == True:
          
                # Write the frame into the
                # file 'filename.avi'
                result.write(frame)
          
                # Display the frame
                # saved in the file
                cv.imshow('Video Capture', frame)
          
                # Press S on keyboard 
                # to stop the process
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
          
            # Break the loop
            else:
                break
          
        # When everything done, release 
        # the video capture and video 
        # write objects
        
        #video.release()
        result.release()

        # Destroy all the windows
        cv.destroyAllWindows()
        self.triger(self.update_cam)


    def save_face(self):
        pass


    # Image Segmentation Filter: Apply image segmentation filter

    def glasses(self, dt):
        
        #Changed to image segmentation using thersholding
        
        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        
        
        
        
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray,
                                             # scaleFactor=1.1,
                                             # minNeighbors=10,
                                             # minSize=(100, 100),
                                             # flags=cv.CASCADE_SCALE_IMAGE)

        # frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

        

        # #print(type(faces))
        # for (x, y, w, h) in faces:
                
            # roi_gray = gray[y:y+h, x:x+h]
            # roi_color = frame[y:y+h, x:x+h]

            # eyes = eyes_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
                # roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
                # glasses2 = image_resize(glasses.copy(), width=ew)

                # gw, gh, gc = glasses2.shape
                # for i in range(0, gw):
                    # for j in range(0, gh):
                        # if glasses2[i, j][3] != 0: # alpha 0
                            # roi_color[ey + i, ex + j] = glasses2[i, j]


            # nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=20)
            # for (nx, ny, nw, nh) in nose:
                # roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
                # mustache2 = image_resize(mustache.copy(), width=nw)

                # mw, mh, mc = mustache2.shape
                # for i in range(0, mw):
                    # for j in range(0, mh):
                        # if mustache2[i, j][3] != 0: # alpha 0
                            # roi_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]

        # # Display the resulting frame
        # frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        
        
        #Apply Smoothing using mean filter
        #Apply mean filter using matrix of ones with dimensions 21*21 divided by the no. of the elements to apply smoothing
        meank=(1/441)*(np.ones((21,21),np.float32))
        frame=cv.filter2D(src=frame, ddepth=-1, kernel=meank)
        #Converting the colored image into gray scale image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #Apply Segmentation using Thresholding
        #Apply mask to R,G,B channels then concatenate them
        def filter_image(frame, mask):
            r = frame[:,:,0] * mask
            g = frame[:,:,1] * mask
            b = frame[:,:,2] * mask
            return np.dstack([r,g,b])
        #Find thresholding factor of gray scale image
        thresh = threshold_otsu(gray)
        #Find pixels with values less than thresholding factor
        img_otsu  = gray < thresh
        filtered = filter_image(frame, img_otsu)    
        
        
        #frame= sobel_grad
        frame = cv.cvtColor(filtered, cv.COLOR_BGRA2BGR)


        #convert frame to texture
        buf = cv.flip(frame, 0)
        buf = buf.tobytes()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # display frame from the texture
        self.ids.cam.texture = texture1 

         # testing
        #if len(faces) != 0:
        	
        	#self.have_face = True
           


    #  Style transfer: Can take a photo from the user and a photo that he/she wishes to merge together to add a 
    #  custom effect or its style to his/her photo.

    def style_transfer(self, dt):


        # grab the paths to all neural style transfer models in our 'models'
        # directory, provided all models end with the '.t7' file extension
        modelPaths = imutils.paths.list_files("models", validExts=(".t7",))
        modelPaths = sorted(list(modelPaths))

        # generate unique IDs for each of the model paths, then combine the
        # two lists together
        models = list(zip(range(0, len(modelPaths)), (modelPaths)))
        print('here')
        # use the cycle function of itertools that can loop over all model
        # paths, and then when the end is reached, restart again
        # modelIter = itertools.cycle(models)
        # (modelID, modelPath) = next(modelIter)
        
        #idx = 1
        #(modelID, modelPath) = models[idx % len(models)]
        #net = cv.dnn.readNetFromTorch(modelPath)


        #print("[INFO] starting video stream...")
        #vs = VideoStream(src = 0).start()
        # load the neural style transfer model from disk
        #print("[INFO] loading style transfer model...")
        #net = cv.dnn.readNetFromTorch(modelPath)
        
        
        idx = 0
        (modelID, modelPath) = models[idx % len(models)]

        # load the neural style transfer model from disk
        print("[INFO] loading style transfer model...")
        net = cv.dnn.readNetFromTorch(modelPath)
        print('here')
        # initialize the video stream, then allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        #vs = VideoStream(src=0).start()
        #time.sleep(2.0)
        print("[INFO] {}. {}".format(modelID + 1, modelPath))



        originalCount, styledCount = 0,0




        ret, frame = self.capture.read()
        print(frame)
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)

        vs = VideoStream(src=0).start()


        while self.styling == True:
                
            # grab the frame from the threaded video stream
            frame = vs.read()
            print(frame)
            # resize the frame to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image dimensions
            frame = imutils.resize(frame, width=600)
            orig = frame.copy()
            (h, w) = frame.shape[:2]

            # construct a blob from the frame, set the input, and then perform a
            # forward pass of the network
            blob = cv.dnn.blobFromImage(frame, 1.0, (w, h),
                (103.939, 116.779, 123.680), swapRB=False, crop=False)
            net.setInput(blob)
            output = net.forward()

            # reshape the output tensor, add back in the mean subtraction, and
            # then swap the channel ordering
            output = output.reshape((3, output.shape[2], output.shape[3]))
            output[0] += 103.939
            output[1] += 116.779
            output[2] += 123.680
            output /= 255.0
            output = output.transpose(1, 2, 0)

            # show the original frame along with the output neural style transfer
            
            cv.imshow("Output", output)

            key = cv.waitKey(1) & 0xFF

            # if the `n` key is pressed (for "next"), load the next neural style transfer model
            if key == ord("n"):
                # (modelID, modelPath) = next(modelIter)
                idx += 1
                (modelID, modelPath) = models[idx % len(models)]
                #cv.putText(frame, str(modelPath.split('/')[-1].split(".")[0]) ,(10,40), cv.FONT_HERSHEY_TRIPLEX, 1, (255,255,255) , 2)
                #cv.imshow("Input", frame)
                cv.waitKey(20)
                print("[INFO] {}. {}".format(modelID + 1, modelPath))
                net = cv.dnn.readNetFromTorch(modelPath)

            # if the `p` key is pressed (for "previous"), load the previous neural style transfer model
            elif key == ord("p"):
                idx -= 1
                (modelID, modelPath) = models[idx % len(models)]
                #cv.putText(frame, str(modelPath.split('/')[-1].split(".")[0]) ,(10,40), cv.FONT_HERSHEY_TRIPLEX, 1, (255,255,255) , 2)
                #cv.imshow("Input", frame)
                cv.waitKey(20)
                print("[INFO] {}. {}".format(modelID + 1, modelPath))
                net = cv.dnn.readNetFromTorch(modelPath)
                

            #save the styled image
            elif key == ord("s"):
                
                output *= 255.0


                DIR = './gallery'
                images = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
                count = len(images)
                while 'capture{}.jpg'.format(count) in images:
                        count+=1

                
                cv.imwrite('gallery/capture{}.jpg'.format(count), output)

##
##                # store as blob in DB
##                with open('gallery/capture{}.jpg'.format(count), "rb") as file:
##                        bData = file.read()
##                
##                sqlStatment = "INSERT INTO capture (photo) VALUES (%s)"
##                cursor.execute(sqlStatment, (bData, ))
##                connection.commit()





            # otheriwse, if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                self.styling = False
                break

            elif cv.getWindowProperty('Output', 0) < 0:
                self.styling = False
                break
                
                
            buf = cv.flip(frame, 0)
            buf = buf.tobytes()
            texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.ids.cam.texture = texture_f

        # do a bit of cleanup
        cv.destroyAllWindows()
        vs.stop()
        self.triger(self.update_cam)


    #  Painter: We use a pencil to write on the screen
    
    def painter(self, dt):
        ret, frame = self.capture.read()
        frame = cv.resize(frame, (1090, 720), interpolation = cv.INTER_AREA)
        frame_w, frame_h, _ = frame.shape
        frame = cv.flip(frame, 1)           #Mirror
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Add the coloring options to the frame
        center_x, center_y = int(frame_w/2.5), int(frame_h/15)
        radius = 60
        space = 2*radius + 40

        #Clear button
        cv.circle(frame, (center_x, center_y), radius, colors[6], -1)
        cv.putText(frame, "CLEAR", (center_x-radius+20, center_y+10), font, .8, colors[4], 2)

        #Blue button
        cv.circle(frame, (center_x+space, center_y), radius, colors[0], -1)
        cv.putText(frame, "BLUE", (center_x+space-radius+25, center_y+10), font, .8, colors[4], 2)

        #Green button
        cv.circle(frame, (center_x+2*space, center_y), radius, colors[1], -1)
        cv.putText(frame, "GREEN", (center_x+2*space-radius+20, center_y+10), font, .8, colors[4], 2)

        #Red button
        cv.circle(frame, (center_x+3*space, center_y), radius, colors[2], -1)
        cv.putText(frame, "RED", (center_x+3*space-radius+35, center_y+10), font, .8, colors[4], 2)

        # Determine which pixels fall within the blue boundaries and then blur the binary image
        blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
        blue_mask = cv.erode(blue_mask, kernel, iterations=2)
        blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
        blue_mask = cv.dilate(blue_mask, kernel, iterations=1)

        # Find contours in the image
        cnts, _ = cv.findContours(blue_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        center = None

        # Check to see if any contours were found
        if len(cnts) > 0:
    	    # Sort the contours and find the largest one -- we
    	    # will assume this contour correspondes to the area of the bottle cap
            cnt = sorted(cnts, key = cv.contourArea, reverse = True)[0]
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv.minEnclosingCircle(cnt)
            # Draw the circle around the contour
            cv.circle(frame, (int(x), int(y)), int(radius), colors[3], 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            global bpoints, gpoints, rpoints, bindex, gindex, rindex, colorIndex
            if center_y-radius <= center[1] <= center_y+radius:
                if center_x-radius <= center[0] <= center_x+radius: # Clear All
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]

                    bindex = 0
                    gindex = 0
                    rindex = 0

                elif center_x+space-radius <= center[0] <= center_x+space+radius:
                        colorIndex = 0 # Blue
                elif center_x+2*space-radius <= center[0] <= center_x+2*space+radius:
                        colorIndex = 1 # Green
                elif center_x+3*space-radius <= center[0] <= center_x+3*space+radius:
                        colorIndex = 2 # Red
            else :
                if colorIndex == 0:
                    bpoints[bindex].appendleft(center)
                elif colorIndex == 1:
                    gpoints[gindex].appendleft(center)
                elif colorIndex == 2:
                    rpoints[rindex].appendleft(center)

        # Append the next deque when no contours are detected (i.e., bottle cap reversed)
        else:
            bpoints.append(deque(maxlen=512))
            bindex += 1
            gpoints.append(deque(maxlen=512))
            gindex += 1
            rpoints.append(deque(maxlen=512))
            rindex += 1

        # Draw lines of all the colors (Blue, Green and Red)
        points = [bpoints, gpoints, rpoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        buf = cv.flip(frame, 0)
        buf = buf.tobytes()
        texture_f = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture_f.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.ids.cam.texture = texture_f

    
    # Update Identifier: save his/her picture in the faces pickle file for further identification

    
    def update_faces_data(self):
        get_faces_data("wb")

        

kv = Builder.load_file("screens.kv")

class FaceAPP(App):
    def build(self):
        Window.size = (1115, 540)
        Window.minimum_width = 1115
        Window.minimum_height = 540
        #Window.clearcolor = (.7, .7, .7, 1)
        #Window.borderless = "1"
        #Window.fullscreen = 'fake'
        #Window.set_system_cursor('size_we')
        Window.softinput_mode = 'resize'
        return kv

FaceAPP().run()
