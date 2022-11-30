# Flappy Bird Computer Vision Project

"""Importing the required Libraries"""

import sys  # to manipulate different parts of the Python runtime environment
import time  # For Time
import random  # For Random values
import pygame  # For Python games
import cv2 as cv  # For Computer Vision
import mediapipe  # face detection
from collections import deque  # To keep track of the pipes

"""This Code will help us recognise and to put a mesh on the face"""
mp_face_recognition = mediapipe.solutions.drawing_utils
mp_face_recognition_styles = mediapipe.solutions.drawing_styles
mp_face_mesh = mediapipe.solutions.face_mesh

"""Drawing specifications"""
thickness = 1
circle_radius = 1
drawing_specifications = mp_face_recognition.DrawingSpec(thickness=thickness, circle_radius=circle_radius)

"""initiating pygame"""
pygame.init()
Video_capturing = cv.VideoCapture(0, cv.CAP_DSHOW)  # Video Capturing From the Camera "0" is the id of Camera
window_game_size = (Video_capturing.get(cv.CAP_PROP_FRAME_WIDTH),
                    Video_capturing.get(cv.CAP_PROP_FRAME_HEIGHT))  # Setting the window game size

game_screen = pygame.display.set_mode(window_game_size)  # Setting pygame Screen

"""Facial Recognition"""
"""MediaPipe Face Mesh is a solution that estimates 468 3D face landmarks in
 accurately around lips, eyes and irises"""


with mp_face_mesh.FaceMesh(max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while True:  # Starting an infinite loop.

        for event in pygame.event.get():  # For stopping the game.

            # Close the game if you quit by ctrl + w or Crossing.
            if event.type == pygame.QUIT:
                Video_capturing.release()
                cv.destroyAllWindows()  # Destroying all Windows Created.
                pygame.quit()  # Quitting pygame.
                sys.exit()  # For exiting the program.

            # Capturing the frames and the return value.
            ret, frame = Video_capturing.read()
            if not ret:  # if returned value is false and no frame is captured.
                print("No Frame Got...")
                continue

            game_screen.fill((150, 150, 150))  # fill with some random color later we will overwrite this.

            """Face Mesh, making our frame writable false so it speeds up the face detection processes"""

            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Converts the BGR frame to RGB
            results = face_mesh.process(frame)
            frame.flags.writeable = True

            """Now To draw the mesh around the face"""

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    """This right here is Drawing the actual mesh on the face"""

                    """For Face mesh tesselation used to locate and detect original face images."""
                    mp_face_recognition.draw_landmarks(image=frame,
                                                       landmark_list=face_landmarks,
                                                       connections=mp_face_mesh.FACEMESH_TESSELATION,
                                                       landmark_drawing_spec=None,
                                                       connection_drawing_spec=mp_face_recognition_styles.get_default_face_mesh_tesselation_style())

                    """for face mesh contours curve joining all the continuous points (along the boundary),"""

                    mp_face_recognition.draw_landmarks(image=frame,
                                                       landmark_list=face_landmarks,
                                                       connections=mp_face_mesh.FACEMESH_CONTOURS,
                                                       landmark_drawing_spec=None,
                                                       connection_drawing_spec=mp_face_recognition_styles.get_default_face_mesh_contours_style())

                    """It also provides a more accurate estimation of the pupil and eye contours which could be used to 
                    detect blinking of eyes."""
                    mp_face_recognition.draw_landmarks(image=frame,
                                                       landmark_list=face_landmarks,
                                                       connections=mp_face_mesh.FACEMESH_IRISES,
                                                       landmark_drawing_spec=None,
                                                       connection_drawing_spec=mp_face_recognition_styles.get_default_face_mesh_iris_connections_style())

                    """Swapping of axis is needed.by mirroring the frame it will be more natural """

                    frame = cv.flip(frame, 1).swapaxes(0, 1)

                    # Putting the frames onto the screen
                    pygame.surfarray.blit_array(game_screen, frame)

                    # Displaying
                    pygame.display.flip()



