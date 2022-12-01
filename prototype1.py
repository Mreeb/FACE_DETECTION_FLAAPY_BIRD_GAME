# Flappy Bird Computer Vision Project

"""Importing the required Libraries"""

import sys  # to manipulate different parts of the Python runtime environment
import time  # For Time
import random  # For Random values
import pygame  # For Python games
import cv2 as cv  # For Computer Vision
import mediapipe  # face detection
from collections import deque  # To keep track of the pipes


class Flappy_Game:
    def __init__(self):

        # image files
        self.window_game_size, self.window_x_axis, self.window_y_axis = None, None, None
        self.game_screen = None
        self.Video_capturing = None
        self.font = "purisa.tff"
        self.color = (100, 255, 255)
        self.image = "ghost_sprite.png"
        self.pipe_image_file = "pipe_sprite_single.png"

        # Pipe initializations
        self.pip_image, self.pipe_frames, self.pipe_starting_template = None, None, None
        # character initializations
        self.frame, self.character, self.character_frame = None, None, None

        self.thickness = 1
        self.circle_radius = 1
        self.camera_id = 0  # Video Capturing From the Camera "0" is the id of Camera
        self.space_between_pipes = 230
        self.divide_factor = 6

        self.Clock, self.Game_Stage, self.Pipe_spawning, self.Pipe_time_diff = None, None, None, None
        self.Pipe_spawn_distance, self.update_Score, self.game_running, self.game_score = None, None, None, None
        self.pipe_speed = None

    def playing_character(self, image):
        self.image = image

    def character_(self):
        self.character = pygame.image.load(self.image, "Game Character")
        width = self.character.get_width() / self.divide_factor
        height = self.character.get_height() / self.divide_factor
        self.character = pygame.transform.scale(self.character, (width, height))
        return self.character

    def pipes(self):
        pass

    def speed(self):
        return self.Pipe_spawn_distance / self.Pipe_time_diff

    def game_settings(self):
        self.Clock, self.Game_Stage, self.Pipe_spawning, self.Pipe_time_diff = time.time(), 1, 0, 35
        self.Pipe_spawn_distance, self.game_score = 450, 0
        self.game_running = True
        self.update_Score = False
        self.pipe_speed = self.speed()

    def game_over_part(self):
        text = pygame.font.SysFont(self.font, 64).render("GAME OVER!", True, self.color)
        text_frame = text.get_rect()
        text_frame.center = (self.window_x_axis / 2, self.window_y_axis / 2)
        self.game_screen.blit(text, text_frame)
        pygame.display.update()
        pygame.time.wait(2000)

    def stages(self):
        pass

    def update_score(self):
        pass

    def Exit(self):
        self.Video_capturing.release()
        cv.destroyAllWindows()  # Destroying all Windows Created.
        pygame.quit()  # Quitting pygame.
        sys.exit()  # For exiting the program.

    def Game_Working(self):
        """This Code will help us recognise and to put a mesh on the face"""
        mp_face_recognition = mediapipe.solutions.drawing_utils
        mp_face_recognition_styles = mediapipe.solutions.drawing_styles
        mp_face_mesh = mediapipe.solutions.face_mesh

        """Drawing specifications"""

        drawing_specifications = mp_face_recognition.DrawingSpec(thickness=self.thickness,
                                                                 circle_radius=self.circle_radius)

        """initiating pygame"""
        pygame.init()

        """Capturing from the Camera"""
        self.Video_capturing = cv.VideoCapture(self.camera_id)
        self.window_game_size = (self.Video_capturing.get(cv.CAP_PROP_FRAME_WIDTH),
                                 self.Video_capturing.get(cv.CAP_PROP_FRAME_HEIGHT))  # Setting the window game size

        self.window_x_axis = self.window_game_size[0]  # Window x-axis
        self.window_y_axis = self.window_game_size[1]  # Window y-axis

        self.game_screen = pygame.display.set_mode(self.window_game_size)  # Setting pygame Screen

        """Character Setting"""
        self.character_frame = self.character_().get_rect()
        x = self.window_x_axis // 6
        y = self.window_y_axis // 2
        self.character_frame.center = (x, y)

        self.pipe_frames = deque()
        self.pip_image = pygame.image.load(self.pipe_image_file, "Pipe Image")
        self.pipe_starting_template = self.pip_image.get_rect()

        self.game_settings()

        """Facial Recognition"""
        """MediaPipe Face Mesh is a solution that estimates 468 3D face landmarks in
         accurately around lips, eyes and irises"""

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5) as face_mesh:
            flag = True
            while flag:  # Starting an infinite loop.

                flag1 = not self.game_running
                if flag1:
                    self.game_over_part()
                    self.Exit()

                for event in pygame.event.get():  # For stopping the game.
                    # Close the game if you quit by ctrl + w or Crossing.
                    if event.type == pygame.QUIT:
                        self.Exit()

                # Capturing the frames and the return value.
                ret, self.frame = self.Video_capturing.read()
                if not ret:  # if returned value is false and no frame is captured.
                    print("No Frame Got...")
                    continue

                self.game_screen.fill((150, 150, 150))  # fill with some random color later we will overwrite this.

                """Face Mesh, making our frame writable false so it speeds up the face detection processes"""

                state = False
                self.frame.flags.writeable = state
                self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)  # Converts the BGR frame to RGB
                results = face_mesh.process(self.frame)
                self.frame.flags.writeable = not state

                """Swapping of axis is needed.by mirroring the frame it will be more natural """

                self.frame = cv.flip(self.frame, 1).swapaxes(0, 1)

                # Putting the frames onto the screen
                pygame.surfarray.blit_array(self.game_screen, self.frame)

                # Displaying
                pygame.display.flip()


if __name__ == "__main__":
    print("Game Starting....")
    obj = Flappy_Game()
    obj.Game_Working()
