# Smart_Face_Identifier_and_Styler


TL;DR:


1- Install requirements.txt file:

pip install -r requirements.txt

2- Then Run main.py


-------------


How to use the application in details:


1. Open cmd in the application directory and type “pip install -r
requirements.txt” to install the required dependencies to run the
application, then type “python main.py” to run it.

2. It opens on the live camera and we can see different features on the
left.

3. If we want to identify the man, Firstly, we choose “Detect Faces” to
draw a rectangle around the face, then press “Identify Faces” which
tells us who this man is by comparing the face with the faces that were
saved in the faces directory.

4. Press “Image Segmentation” if you want to divide the image into its
components (edge detection).

5. It’s simple if you want to write on the screen by pressing the “Painter”
button, then choose which color you want to write with, but we need
to use any object that has a blue color end.

6. “Capture Screen” to save a picture from the stream in the gallery
directory.

7. “Capture Video” to save a video from a stream in the gallery directory
after pressing the “q” button on the keyboard.

8. To merge between two images one from the live stream and the
second from saved styles, press “Style Transfer”. From the keyboard, if
we

A. Press “n” to load the next neural style transfer model.
B. Press “p” to load the previous neural style transfer model.
C. Press “s” to save the resulting image to the gallery directory.

9. “Clear” button to delete any feature we have used.

10. To add a new person we should first create a directory manually with
his name and add his picture, then press “Update Identifier”.

11. Press “Esc” on the keyboard to exit from the application.
