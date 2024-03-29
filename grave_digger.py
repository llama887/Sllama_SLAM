from vis_nav_game import Player, Action, Phase
import pygame
import cv2
from typing import List, Any, Tuple
import numpy as np

from deadreckoning import Localizer
from place_recognition import Target_Locator, Image_Embedding

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        self.localizer = Localizer()  # Dead reckoning localizer
        self.target_locator = Target_Locator()
        self.previous_pose : Tuple[int, int, int]= (None, None, None)
        self.target = []
        super(KeyboardPlayerPyGame, self).__init__()

        self.key_hold_state = {
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LSHIFT: False,
        }
        self.is_navigation = False

    def reset(self) -> None:
        self.is_navigation = False
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Reset location
        self.localizer = Localizer()
        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_p: 1,
            pygame.K_r: 1,
            pygame.K_t: 1,
            pygame.K_LSHIFT: 1,
        }
        
    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.K_RETURN:
                # train vocabulary when done exploring
                self.last_act = Action.IDLE
                self.post_exploration()
                return Action.IDLE
            if event.type == pygame.KEYDOWN:
                self.key_hold_state[event.key] = True
                if event.key in self.keymap and self.keymap[event.key] != 1: # 1 is a placeholder keymapping for custom key mappings
                    self.last_act |= self.keymap[event.key]
                elif event.key not in self.keymap:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                self.key_hold_state[event.key] = False
                if event.key in self.keymap and self.keymap[event.key] != 1:
                    self.last_act ^= self.keymap[event.key]
        if not self.key_hold_state[pygame.K_LSHIFT]: # BUG: LSHIFT causes mapping to set player to 0, 0
            self.localizer.track(self.key_hold_state)
            self.localizer.map.update_minimap(self.localizer.current_x, self.localizer.current_y)
        return self.last_act
    
    def post_exploration(self) -> None:
        self.is_navigation = True
        self.target_locator.generate_vocabulary()
    def show_target_images(self):
        self.localizer.current_x = 0
        self.localizer.current_y = 0
        self.localizer.heading = 0
        targets = self.get_target_images()

        best_indexes = []
        for target in targets:
            if self.target_locator.visual_dictionary is None:
                # train dictionary if there is none
                self.target_locator.generate_vocabulary()
                if self.target_locator.visual_dictionary is None:
                    raise ValueError("Visual dictionary is not generated yet")
            best_indexes.append(self.target_locator.find_target_indices(target))
        
        if targets is None or len(targets) <= 0:
            return

                # Concatenate best match images in pairs horizontally and then vertically
        hor1 = cv2.hconcat(
            [self.target_locator.embeddings[best_indexes[0][0]].image, self.target_locator.embeddings[best_indexes[1][0]].image]
        )
        hor2 = cv2.hconcat(
            [self.target_locator.embeddings[best_indexes[2][0]].image, self.target_locator.embeddings[best_indexes[3][0]].image]
        )
        concat_img = cv2.vconcat([hor1, hor2])

        # Concatenate second best match images similarly as above
        hor1_second_best = cv2.hconcat(
            [self.target_locator.embeddings[best_indexes[0][1]].image, self.target_locator.embeddings[best_indexes[1][1]].image]
        )
        hor2_second_best = cv2.hconcat(
            [self.target_locator.embeddings[best_indexes[2][1]].image, self.target_locator.embeddings[best_indexes[3][1]].image]
        )
        concat_img_second_best = cv2.vconcat([hor1_second_best, hor2_second_best])

        # Concatenate target images similarly as above
        hor1_target = cv2.hconcat(targets[:2])
        hor2_target = cv2.hconcat(targets[2:])
        concat_img_target = cv2.vconcat([hor1_target, hor2_target])

        # Concatenate third best match images similarly as above
        hor1_third_best = cv2.hconcat(
            [self.target_locator.embeddings[best_indexes[0][2]].image, self.target_locator.embeddings[best_indexes[1][2]].image]
        )
        hor2_third_best = cv2.hconcat(
            [self.target_locator.embeddings[best_indexes[2][2]].image, self.target_locator.embeddings[best_indexes[3][2]].image]
        )
        concat_img_third_best = cv2.vconcat([hor1_third_best, hor2_third_best])

        # Get width and height for text placement before scaling
        w, h = concat_img_target.shape[:2]

        # Settings for text
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        color = (0, 0, 0)

        # Scaling factor for the images
        scale_factor = 2
        text_scale_factor = 1.2

        # Resize images with scale factor
        concat_img = cv2.resize(concat_img, (0, 0), fx=scale_factor, fy=scale_factor)
        concat_img_second_best = cv2.resize(
            concat_img_second_best, (0, 0), fx=scale_factor, fy=scale_factor
        )
        concat_img_third_best = cv2.resize(
            concat_img_third_best, (0, 0), fx=scale_factor, fy=scale_factor
        )
        concat_img_target = cv2.resize(
            concat_img_target, (0, 0), fx=scale_factor, fy=scale_factor
        )

        # Settings for text after scaling
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        color = (0, 0, 255)  # Red color for visibility

        # Scale offsets and stroke for the scaled image
        scaled_w_offset = int(w_offset * scale_factor)
        scaled_h_offset = int(h_offset * scale_factor)
        scaled_font_size = size * text_scale_factor
        scaled_stroke = int(stroke * scale_factor)

        # Calculate positions for text based on scaled image size
        position_front_view = (scaled_w_offset, scaled_h_offset)
        position_right_view = (
            int(concat_img_target.shape[1] / 2) + scaled_w_offset,
            scaled_h_offset,
        )
        position_back_view = (
            scaled_w_offset,
            int(concat_img_target.shape[0] / 2) + scaled_h_offset,
        )
        position_left_view = (
            int(concat_img_target.shape[1] / 2) + scaled_w_offset,
            int(concat_img_target.shape[0] / 2) + scaled_h_offset,
        )

        # Place text for views on the target image
        cv2.putText(
            concat_img_target,
            "Front View",
            position_front_view,
            font,
            scaled_font_size,
            color,
            scaled_stroke,
            line,
        )
        cv2.putText(
            concat_img_target,
            "Right View",
            position_right_view,
            font,
            scaled_font_size,
            color,
            scaled_stroke,
            line,
        )
        cv2.putText(
            concat_img_target,
            "Back View",
            position_back_view,
            font,
            scaled_font_size,
            color,
            scaled_stroke,
            line,
        )
        cv2.putText(
            concat_img_target,
            "Left View",
            position_left_view,
            font,
            scaled_font_size,
            color,
            scaled_stroke,
            line,
        )

        # Now, apply the text with correct scaling and positioning
        for index in range(1, 5):  # Loop through the indexes
            for rank in range(
                3
            ):  # Loop through the ranks: best, second best, third best
                # Choose the correct image based on rank
                if rank == 0:
                    image_to_draw = concat_img
                elif rank == 1:
                    image_to_draw = concat_img_second_best
                else:
                    image_to_draw = concat_img_third_best

                # Calculate the position for the text based on the image quadrant
                x_offset = (
                    h_offset if (index % 2) != 0 else int(w / 2) + h_offset
                )  # Left if 1 or 3, right if 2 or 4
                y_offset = (
                    w_offset if index < 3 else int(w / 2) + w_offset
                )  # Top if 1 or 2, bottom if 3 or 4

                # Draw the text with the scaled positions and sizes
                cv2.putText(
                    image_to_draw,
                    f"Selection: {3*(index-1) + rank + 1}\t\t",
                    (x_offset * scale_factor, y_offset * scale_factor),  # Scaled offset
                    font,
                    scaled_font_size,
                    color,
                    scaled_stroke,
                    line,
                )
        # Concatenate the images again for the final display
        top_row = cv2.hconcat([concat_img, concat_img_second_best])
        bottom_row = cv2.hconcat([concat_img_third_best, concat_img_target])

        # Create and resize window for display
        cv2.namedWindow(
            "KeyboardPlayer:targets and recognized", cv2.WINDOW_NORMAL
        )  # Create a resizable window
        cv2.resizeWindow(
            "KeyboardPlayer:targets and recognized", top_row.shape[1], top_row.shape[0]
        )  # Set the window size

        # Display the image
        cv2.imshow(
            "KeyboardPlayer:targets and recognized", cv2.vconcat([top_row, bottom_row])
        )
        cv2.waitKey(1)
        return best_indexes
    
    def show_target_as_reference(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)
    
    def set_target_images(self, images):
        """
        Set target images
        """
        self.post_exploration()
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.target = np.ravel(self.show_target_images())
        if self.target is None or len(self.target) <= 1:
            return
        target_index = (int(input(f"Enter the row index (between 0 and {len(self.target) - 1}): "))- 1)
        target_x = self.target_locator.embeddings[self.target[target_index]].x
        target_y = self.target_locator.embeddings[self.target[target_index]].y
        self.localizer.map.target = (target_x, target_y)    
        cv2.destroyAllWindows()
        self.show_target_as_reference()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')
        print("When done exploring, press Enter to start training the visual dictionary.")

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        #print(f"Current position: {self.localizer.get_pose()}, Previous position: {self.previous_pose}, Navigation: {self.is_navigation}")
        if not self.is_navigation and self.localizer.get_pose() != self.previous_pose:
            self.previous_pose = self.localizer.get_pose()
            #print("Embedding image ...")
            self.target_locator.add_image(Image_Embedding(self.localizer.get_pose(), self.fpv))

        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())