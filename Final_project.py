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
        self.color = (0, 0, 0)
        self.image = "black_ghost_sprite.png"
        self.pipe_image_file = "Pipe.png"

        # Pipe initializations
        self.pip_image, self.pipe_frames, self.pipe_starting_template = None, None, None
        # character initializations
        self.frame, self.character, self.character_frame = None, None, None

        self.thickness = 1
        self.circle_radius = 1
        self.camera_id = 0  # Video Capturing From the Camera "0" is the id of Camera
        self.space_between_pipes = 250
        self.divide_factor = 6

        self.Clock, self.Game_Stage, self.Pipe_spawning, self.Pipe_time_diff = None, None, None, None
        self.Pipe_spawn_distance, self.update_Score, self.game_running, self.game_score = None, None, None, None
        self.game_points, self.pipe_speed, self.timer = None, None, None

        self.detection_position = 94

    def playing_character(self, image):
        self.image = image

    def character_(self):
        self.character = pygame.image.load(self.image, "Game Character")
        width = self.character.get_width() / self.divide_factor
        height = self.character.get_height() / self.divide_factor
        self.character = pygame.transform.scale(self.character, (width, height))
        return self.character

    # def del_pipes_left(self):
    #     len_pipe_frames = len(self.pipe_frames)
    #     right_pipes = self.pipe_frames[0][0].right
    #     if len_pipe_frames > 0 and right_pipes < 0:
    #         self.pipe_frames.popleft()

    def speed(self):
        return self.Pipe_spawn_distance / self.Pipe_time_diff

    def game_settings(self):
        self.Clock, self.Game_Stage, self.Pipe_spawning, self.Pipe_time_diff = time.time(), 1, 0, 40
        self.Pipe_spawn_distance, self.game_score = 500, 0
        self.game_running = True
        self.update_Score = False
        self.pipe_speed = self.speed()
        self.game_points = 1
        self.timer = 1

    def game_over_part(self):
        self.color = (255,0,0)
        text = pygame.font.SysFont(self.font, 64).render("GAME OVER!", True, self.color)
        text_frame = text.get_rect()
        text_frame.center = (self.window_x_axis / 2, self.window_y_axis / 2)
        self.game_screen.blit(text, text_frame)
        pygame.display.update()
        pygame.time.wait(2000)

    def stage_text(self):
        text = pygame.font.SysFont(self.font, 25).render(f'Stage: {self.Game_Stage}', True, self.color)
        text_rect = text.get_rect()
        text_rect.center = (50, 25)
        self.game_screen.blit(text, text_rect)

    def score_text(self):
        text = pygame.font.SysFont(self.font, 25).render(f'Score: {self.game_score}', True, self.color)
        text_rect = text.get_rect()
        text_rect.center = (50, 50)
        self.game_screen.blit(text, text_rect)

    def update_score(self):
        pass

    def timings(self):
        self.Pipe_time_diff *= 5 / 6
        self.Game_Stage += 1
        self.Clock = time.time()

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
        self.character = pygame.image.load(self.image, "Game Character")
        width = self.character.get_width()/self.divide_factor
        height = self.character.get_height()/self.divide_factor
        self.character = pygame.transform.scale(self.character, (width, height))

        self.character_frame = self.character.get_rect()
        x = self.window_x_axis // 6
        y = self.window_y_axis // 2
        self.character_frame.center = (self.window_x_axis // 6, self.window_y_axis // 2)

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

                """Bird Position Adjustments"""

                # res = results.multi_face_landmarks
                # res_len = len(results.multi_face_landmarks)
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:

                    marker = results.multi_face_landmarks[0].landmark[self.detection_position].y
                    self.character_frame.centery = +self.window_y_axis/2 + (marker - 0.5) * 1.5 * self.window_y_axis

                    if self.character_frame.top < 0:
                        self.character_frame.y = 0

                    if self.character_frame.bottom > self.window_y_axis:
                        self.character_frame.y = self.window_y_axis - self.character_frame.height

                """Swapping of axis is needed.by mirroring the frame it will be more natural """

                self.frame = cv.flip(self.frame, 1).swapaxes(0, 1)

                """Update the pipes"""

                for pipe_frame in self.pipe_frames:
                    pipe_frame[0].x -= self.speed()
                    pipe_frame[1].x -= self.speed()

                # self.del_pipes_left()

                len_pipe_frames = len(self.pipe_frames)
                if len_pipe_frames > 0 and self.pipe_frames[0][0].right < 0:
                    self.pipe_frames.popleft()

                """Update screen"""
                # Putting the frames onto the screen

                pygame.surfarray.blit_array(self.game_screen, self.frame)
                self.game_screen.blit(self.character, self.character_frame)
                counter = True
                for pipe_frame in self.pipe_frames:
                    # Check if bird went through to update score

                    if pipe_frame[0].left <= self.character_frame.x <= pipe_frame[0].right:
                        counter = False

                        if not self.update_Score:
                            self.game_score += self.game_points
                            self.update_Score = True
                    # Update screen

                    self.game_screen.blit(self.pip_image, pipe_frame[1])
                    self.game_screen.blit(pygame.transform.flip(self.pip_image, 0, 1), pipe_frame[0])

                if counter:
                    self.update_Score = False

                """Stage and Score Text"""
                self.stage_text()
                self.score_text()

                pygame.display.flip()

                if any([self.character_frame.colliderect(pipe_frame[0]) or
                        self.character_frame.colliderect(pipe_frame[1])
                        for pipe_frame in self.pipe_frames]):
                    self.game_running = False

                if self.Pipe_spawning == 0:
                    top = self.pipe_starting_template.copy()
                    top.x = random.randint(self.window_x_axis-100, self.window_x_axis)
                    top.y = random.randint(110 - 1000, self.window_y_axis - 120 - self.space_between_pipes - 1000) + random.randint(-100, 50)

                    bottom = self.pipe_starting_template.copy()
                    bottom.x = random.randint(self.window_x_axis-100, self.window_x_axis)
                    bottom.y = top.y + random.randint(900, 1000) + self.space_between_pipes

                    # Appending the Pipes
                    self.pipe_frames.append([top, bottom])

                self.Pipe_spawning += self.timer
                if self.Pipe_spawning >= self.Pipe_time_diff:
                    self.Pipe_spawning = 0

                # Update stage
                if time.time() - self.Clock >= 10:
                    # self.Pipe_time_diff *= 5 / 6
                    # self.Game_Stage += 1
                    # self.Clock = time.time()

                    self.timings()
                # Displaying


if __name__ == "__main__":
    print("Game Starting....")
    obj = Flappy_Game()
    obj.Game_Working()