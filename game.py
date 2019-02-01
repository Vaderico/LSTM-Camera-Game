import contextlib
with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import os
import shutil
import json
import random
import datetime
import time
import csv

from player import HumanPlayer, LSTMPlayer
 
# Game that displays a crop of an image, and allows a player to move
# around it by supplying velocities in the x and y direction
class Game: 
    def __init__(self, player, images_dir, res, frame_rate):
        # Player that supplies x and y velocitites
        self._player = player 
        # Directory of all images used in the game
        self._images_dir = images_dir
        # Frame rate of game
        self._frame_rate = frame_rate
        # width and height resolution that image frame gets stretched out to for display
        self._res = res
        # Boolean to end game loop
        self._running = True

        # Image surface that gets rendered each frame
        self._display_surf = None 
        # Text in the top window bar
        self._caption = "LSTM Camera Game"
        # Height of game entire window
        self._scr_height = res

        # Meta data from the images directory
        self._meta_data = {}
        # Name of meta data file
        self._meta_name = "meta.json"

        # Paths of all complete images that the game cycles through using the RETURN key
        self._all_images = []
        # Index of current image to load from _all_images
        self._img_index = 0
        # Current loaded image from _all_images 
        self._full_image = None
        # Width of _full_image
        self._full_width = 0
        # Height of _full_image
        self._full_height = 0

        # Time of start recording, used for the filenames
        self._time = ""
        # Width and height of frame that is cropped from _full_image to be displayed
        self._frame_size = 224
        # Cropped image of _full_image
        self._small_frame = None
        # Frame that gets displayed - _small_frame expanded to the dimensions of _res
        self._frame = None

        # Top-left x and y coordinate of _small_frame, that has been cropped from _full_image
        self._x_pos = 0
        self._y_pos = 0

    # Initialises pygame and other dependant variables
    def on_init(self):
        # Create pygame window
        pygame.init()
        # Ensure loop is running
        self._running = True
        # Set caption at the top of window
        pygame.display.set_caption(self._caption)

        # Load metadata json from file
        with open (self._images_dir + self._meta_name) as file:
            self._meta_data = json.load(file)
        self._all_images = self._meta_data["test_images"]

        # Create the pygame window with the correct dimensions
        self._display_surf = pygame.display.set_mode((self._res, self._scr_height), 
                pygame.HWSURFACE)
        # Load the image and crop a frame for display
        self.load_image()

    # Go to next image, and load the frame
    def load_next_image(self):
        self.next_image()
        self.load_image()
 
    # Update image index to point to the next image
    def next_image(self):
        self._img_index = (self._img_index + 1) % len(self._all_images)

    # Load the image and crop a frame for display
    def load_image(self):
        # Reset state of player
        self._player.reset()
        # Gets the path of the current image and load it
        image_path = self._images_dir + self._all_images[self._img_index]
        self._full_image = pygame.image.load(image_path)

        # Extract dimensions of loaded image
        self._full_width, self._full_height = self._full_image.get_size()

        # Choose a random starting position for the frame
        self._x_pos = random.randint(0, self._full_width - self._frame_size)
        self._y_pos = random.randint(0, self._full_height - self._frame_size)
        # choose = random.randint(0,3)
        # offset = random.randint(0,9)
        # if choose is 0:
            # self._x_pos = 0 + offset
        # elif choose is 1:
            # self._x_pos = self._full_width - self._frame_size - offset
        # elif choose is 2:
            # self._y_pos = 0 + offset
        # elif choose is 3:
            # self._y_pos = self._full_height - self._frame_size - offset

        # Crops and loads the new frame
        self.update_frame(0, 0)

    # Crops and loads frame for display
    def update_frame(self, dx, dy):
        # Step x and y coordinates with current dx and dy velocities
        new_x = self._x_pos + dx
        new_y = self._y_pos + dy
        # Max x and y to ensure frame doesnt move off the screen
        max_x = self._full_width - self._frame_size
        max_y = self._full_height - self._frame_size
        # If x or y has moved off the screen, keep it in the screen
        self._x_pos = min(max(new_x, 0), max_x)
        self._y_pos = min(max(new_y, 0), max_y)

        # If the frame reaches the edge of the image, reset the player velocities to 0
        if self._x_pos == 0 or self._x_pos == max_x:
            self._player.reset_x_velocity()
        if self._y_pos == 0 or self._y_pos == max_y:
            self._player.reset_y_velocity()

        # Create an empty image to load the cropped frame to
        self._small_frame = pygame.Surface((self._frame_size, self._frame_size))

        # Calculate the crop bounding box for frame
        xstart = self._x_pos
        xend = self._x_pos + self._frame_size
        ystart = self._y_pos
        yend = self._y_pos + self._frame_size

        # Crop frame from larger image
        self._small_frame.blit(self._full_image, (0,0), (xstart, ystart, xend, yend))
        # Blow up small crop to fit the dimensions of the screen
        self._frame = pygame.transform.scale(self._small_frame, (self._res, self._res)).convert()

    # Action to run for each frame cycle
    def on_loop(self):
        # Get the next velocities from player
        dx, dy = self._player.step_velocity(self._small_frame)
        # Load the next frame
        self.update_frame(dx, dy)

    # Render runs each game loop cycle
    def on_render(self):
        # Copy frame to surface, and display on screen
        self._display_surf.blit(self._frame,(0,0))
        pygame.display.flip()

    # Handle events specific to base Game class
    def handle_event(self, event):
        if event.type == KEYDOWN:
            if event.key == K_RETURN:
                # RETURN key pressed, move to next image
                self.load_next_image()

    # Event handler for every single event called
    def on_event(self, event):
        if event.type == QUIT:
            # Exit the program
            self._running = False

        # Pass the event to the player
        self._player.handle_event(event)
        # Call class-specific event handler
        self.handle_event(event)
 
    # End game, cleanup all loose ends
    def on_cleanup(self):
        # Shut down pygame window
        pygame.quit()
 
    # Create game window, and run the game loop
    def on_execute(self):
        # Initialise window
        if self.on_init() == False:
            self._running = False

        # Initialise frame clock
        time_since_action = 1000
        clock = pygame.time.Clock()
 
        # Game Loop
        while (self._running):
            # Call on_event for every current event
            for event in pygame.event.get():
                self.on_event(event)
            
            # Increment frame clock
            dt = clock.tick()
            time_since_action += dt
            # print(1000 / self._frame_rate)

            # If 50 ms has passed, compute/render frame
            if time_since_action >= (1000 / self._frame_rate):
                time_since_action = 0
                self.on_loop()
                self.on_render()

        # Shut down game window
        self.on_cleanup()

# Game that displays a crop of an images, and allows a player to move
# around it by supplying velocities in the x and y direction. Allows user
# to record frames as training data, and shows statistics of recorded frames.
class RecordGame(Game):
    def __init__(self, player, images_dir, training_dir, res, frame_rate, stop, max_frames):
        # Call parent constructor
        Game.__init__(self, player, images_dir, res, frame_rate)
        # Allows recordings to be stopped manually with spacebar
        self._stop_recording = stop
        # Saves path of directory to save training data to 
        self._training_dir = training_dir
        # Name of directory to save temporary images to
        self._temp_dir = self._training_dir + "temp/"
        self._max_frames = max_frames

        # Text in the top window bar
        self._caption = "LSTM Camera Game Recorder"
        # Font style for text display
        self._font_style = "roboto"
        # Height of control panel underneath frame cropping
        self._control_height = 100
        # Colors used for control panel
        self._color_white = (255,255,255)
        self._color_red = (200,50,50)
        self._color_blue = (50,50,200)
        self._color_black = (40,40,40)
        # Fonts with different font sizes for Control panel
        self._font_title = None
        self._font_sub_title = None
        self._font_regular = None
        self._font_small = None

        # Clock to keep track of current time
        self._clock = None
        # Current state of GUI: { 0: NORMAL, 1: RECORDING, 2: SAVING }
        self._recording_state = 0

        # Current frame number for the current sequence being recorded
        self._frame_count = 0
        # Time passed for the current sequence recording in milliseconds
        self._record_time = 0
        # Paths of each saved image .jpg and velocities .json files for the current sequence
        self._recorded_velocities = []

        # Number of sequences saved for the current image at _img_index
        self._curr_img_seq_count = []
        # Combined run time of all sequences saved for the current image at _img_index
        self._curr_img_time = []
        # Number of frames for all sequences saved for the current image at _img_index
        self._curr_img_frames = []

        # Total number of sequences saved for all images in _all_images
        self._tot_seq_count = 0
        # Total time for all sequences saved for all images in _all_images
        self._tot_time = 0
        # Total number of frames for all sequences saved for all images in _all_images
        self._tot_frames = 0

        # Stores data about all objects recorded so far
        self._training_meta_data = {}
            
    # Initialises pygame and other dependant variables
    def on_init(self):
        # Sets the height of the screen to include control panel height
        self._scr_height = self._res + self._control_height
        # Run parent init function
        Game.on_init(self)

        # Load all training images to record from
        self._all_images = self._meta_data["training_images"]
        # Load the current image
        self.load_image()
        # Initialise variables for data recording
        self.init_recording_data()

        # Create fonts to be displayed in control panel
        self._font_title = pygame.font.SysFont(self._font_style, 50)
        self._font_sub_title = pygame.font.SysFont(self._font_style, 38)
        self._font_regular = pygame.font.SysFont(self._font_style, 30)
        self._font_small = pygame.font.SysFont(self._font_style, 25)

    # Initialise variables for data recording
    def init_recording_data(self):
        # Total number of images to record from
        num_train = len(self._all_images)
        # Initialse stat arrays to hold the current images stats in its index
        self._curr_img_seq_count = [0] * num_train
        self._curr_img_time = [0] * num_train
        self._curr_img_frames = [0] * num_train

    # Delete the temp file holding the last recorded sequence
    def clear_current_sequence(self):
        # Delete temporary directory holding recorded frames
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    # Adds the information of the previous recording to the total recording information
    def update_meta_data(self):
        img_idx = self._img_index

        # Update current image and total number of sequences
        self._curr_img_seq_count[img_idx] = self._curr_img_seq_count[img_idx] + 1
        self._tot_seq_count += 1

        # Update current image and total sequence run time
        self._curr_img_time[img_idx] = self._curr_img_time[img_idx] + self._record_time
        self._tot_time += self._record_time

        # Update current image and total number of frames
        self._curr_img_frames[img_idx] = self._curr_img_frames[img_idx] + self._frame_count
        self._tot_frames += self._frame_count

        # Add updates stats to metadata json object
        self._training_meta_data["total sequence count"] = self._tot_seq_count
        self._training_meta_data["total time"] = self._tot_time
        self._training_meta_data["total frame count"] = self._tot_frames
        self._training_meta_data["image sequence count"] = self._curr_img_seq_count
        self._training_meta_data["image time"] = self._curr_img_time
        self._training_meta_data["image frame count"] = self._curr_img_frames

    # Saves the metadata for the current sequence, and load it to the JSON file
    def save_current_sequence(self):
        # Save recording information
        self.update_meta_data()

        # Rename temp directory
        if os.path.exists(self._temp_dir):
            os.rename(self._temp_dir, self._training_dir + self._time)

        # Write recorded velocities to file
        with open("%s%s/velocities.csv" % (self._training_dir, self._time), 'w') as csv_file:  
            csv.writer(csv_file, delimiter=',').writerows(self._recorded_velocities)

    # Saved option has been pressed, either delete or save the recorded sequence
    def save_option_pressed(self, is_saving):
        # If the current Game state is not 2: SAVING, exit
        if not self._recording_state == 2:
            return

        # SAVE option chosen
        if is_saving:
            # Save the recorded sequence metadata
            self.save_current_sequence()
            # Move image to the next training image
            self.load_next_image()
        else:
            self.clear_current_sequence()

        # Reset all information saved for sequence
        self._frame_count = 0
        self._record_time = 0
        self._recorded_velocities = []

        # Move state of program to 0: NORMAL
        self._recording_state = 0

    # Update frame coordinates and save frame/velocities
    def on_loop(self):
        # Get the next velocities for current frame
        dx, dy = self._player.step_velocity(self._small_frame)

        # If in recording state, save the current frame and velocities
        if self._recording_state == 1:
            # Time since last frame taken
            dt = self._clock.tick()
            # Append time dince last frame to recorded time
            self._record_time += dt
            # Add 1 to frame count for current sequence recording
            self._frame_count += 1

            # Stop recording when max frames is reached
            if self._frame_count == self._max_frames:
                self.update_recording_state()

            # Filenames for current image and velocities being saved
            frame_filename = "%06d.png" % self._frame_count
            frame_path = self._temp_dir + frame_filename

            # Save current frame and velocities
            pygame.image.save(self._small_frame, frame_path)
            self._recorded_velocities.append([dx, dy])

        # Load the next frame
        self.update_frame(dx, dy)

    # Draw Control panel
    def draw_control_panel(self):
        state = self._recording_state
        # Draw control panel in NORMAL state
        if state == 0:
            self.draw_control_home()
        # Draw background for control panel in RECORDING state
        if state == 1:
            self.draw_control_record()
        # Draw background for control panel in SAVING state
        if state == 2:
            self.draw_control_save()
        # Draw text in control panel in RECORDING and SAVING states
        if state == 1 or state == 2:
            self.draw_record_save_text()

    # Draw control panel in NORMAL state
    def draw_control_home(self):
        # Draw white rectangle for control panel background
        panel_rect = Rect((0,self._res), (self._res, self._control_height))
        pygame.draw.rect(self._display_surf, self._color_white, panel_rect)

        img_idx = self._img_index

        # y coordinate for the top of the text
        y = self._res + 10
        # x coordinate for text on the left hand side of the screen
        left_x = 60
        # x coordinate for text on the right hand side of the screen
        right_x = self._res - 250
        # The amount of space between each line of text
        space = 20
        # Color of the text
        color = self._color_black

        # Draw current image number
        label_str = "Current image: %d/%d" % (img_idx + 1, len(self._all_images))
        self.draw_text(label_str, self._font_small, color, left_x, y)

        y += space
        # Draw current and total image sequence count
        label_str = "Current image sequences: %d" % self._curr_img_seq_count[img_idx]
        self.draw_text(label_str, self._font_small, color, left_x, y)
        label_str = "Total sequences: %d" % self._tot_seq_count
        self.draw_text(label_str, self._font_small, color, right_x, y)

        y += space
        # Draw current and total sequence times
        label_str = "Current image time: %s" % self.ms_to_time(self._curr_img_time[img_idx])
        self.draw_text(label_str, self._font_small, color, left_x, y)
        label_str = "Total time: %s" % self.ms_to_time(self._tot_time)
        self.draw_text(label_str, self._font_small, color, right_x, y)

        y += space
        # Draw current and total number of frames
        label_str = "Current image frames: %d" % self._curr_img_frames[img_idx]
        self.draw_text(label_str, self._font_small, color, left_x, y)
        label_str = "Total frames: %d" % self._tot_frames
        self.draw_text(label_str, self._font_small, color, right_x, y)

    # Draw background for control panel in RECORDING state
    def draw_control_record(self):
        # Draw red rectangle for control panel background
        panel_rect = Rect((0,self._res), (self._res, self._control_height))
        pygame.draw.rect(self._display_surf, self._color_red, panel_rect)

        self.draw_text("Recording", self._font_title, self._color_white, 30, self._res + 30)

    # Draw background for control panel in SAVING state
    def draw_control_save(self):
        # Draw blue rectangle for control panel background
        panel_rect = Rect((0,self._res), (self._res, self._control_height))
        pygame.draw.rect(self._display_surf, self._color_blue, panel_rect)

        label_str = "Save Recording? (Y/N)"
        self.draw_text(label_str, self._font_sub_title, self._color_white, 30, self._res + 35)

    # Draw background for control panel in SAVING state
    def draw_record_save_text(self):
        color = self._color_white
        x = self._res - 260

        label_str = "Time elapsed: %s" % self.ms_to_time(self._record_time)
        self.draw_text(label_str, self._font_regular, color, x, self._res + 20)

        label_str = "Frames: %d" % self._frame_count
        self.draw_text(label_str, self._font_regular, color, x, self._res + 60)

    # Draw text in control panel in RECORDING and SAVING states
    def draw_text(self, label_str, font, color, x, y):
        label = font.render(label_str, True, color)
        self._display_surf.blit(label, (x, y))

    # Convert milliseconds to HH:MM:SS string format
    def ms_to_time(self, ms):
        s = 60
        q = 1000
        return "%02d:%02d:%02d" % (ms/s/s/q, ms/s/q, ms/q)

    # Draw control panel and game to the screen
    def on_render(self):
        Game.on_render(self)
        self.draw_control_panel()

    # Reset directory to store temporary image files
    def refresh_temp_dir(self):
        # Delete directory if path exists
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

        # Create temporary image directory
        os.makedirs(self._temp_dir)

    # Spacebar event, start or stop recording
    def update_recording_state(self):
        # Start recording
        if self._recording_state == 0:
            self._record_time = 0
            self._clock = pygame.time.Clock()
            self._recording_state = 1
            self.refresh_temp_dir()
            self._time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        # Stop recording
        elif self._recording_state == 1:
            self._recording_state = 2

    # Handle events for this class
    def handle_event(self, event):
        if event.type == KEYDOWN:
            if event.key == K_SPACE: 
                # Start or stop recording
                if not (self._recording_state == 1 and not self._stop_recording):
                    self.update_recording_state()
            if event.key == K_RETURN:
                # Load next image if in NORMAL state
                if self._recording_state == 0:
                    self.load_next_image()
            if event.key == K_y:
                # Save recorded sequence
                self.save_option_pressed(True)
            if event.key == K_n:
                # Don't save recorded sequence
                self.save_option_pressed(False)

