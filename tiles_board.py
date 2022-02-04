from tile_detector import TilesDetector
from tile_classifier import TilesClassifier
from game.carcassonne_game_state import CarcassonneGameState
from tile_position_finder import TilePositionFinder
import numpy as np
from PIL import Image
import os

class FailedToRecognizeError(Exception):
    pass

class CarcassoneBoard:
    def __init__(self, det_model_path, cls_model_path):
        self.board_detector = TilesDetector(det_model_path)
        self.tile_classifier = TilesClassifier(cls_model_path)
        self.tiles_locations = []
        self.meep_locations = []
        self.tiles_classes = []
        self.game_state = CarcassonneGameState()

    def recognize_game_situation(self, img: Image):
        tiles_imgs, self.tiles_locations, self.meep_locations = self.board_detector.detect(img)
        self.tiles = [self.tile_classifier.classify(tile) for tile in tiles_imgs]
        for tile, location in zip(self.tiles, self.tiles_locations):
            self.game_state.add_tile(location[0], location[1], tile)

        return self.draw()

    def recognize_tile(self, tile_img: Image):
        tile_img, _, _ = self.board_detector.detect(tile_img)
        if len(tile_img) != 1:
            return None
        tile = self.tile_classifier.classify(tile_img[0])
        return tile

    def draw(self):
        min_y, min_x, max_y, max_x = self.game_state.get_min_max_coord()
        tile_size = 64
        width = (max_x - min_x + 1) * tile_size
        height = (max_y - min_y + 1) * tile_size
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(min_y, max_y + 1):
            for j in range(min_x, max_x + 1):
                if self.game_state.board[i][j] is not None:
                    coord_x = j - min_x
                    coord_y = i - min_y
                    tile = self.game_state.board[i][j]
                    tile_img = Image.open(tile.image)
                    tile_img = np.asarray(tile_img.resize((tile_size, tile_size)))
                    img[coord_y * 64:(coord_y + 1) * tile_size,
                    coord_x * tile_size:(coord_x + 1) * tile_size] = tile_img

        return img

    def get_possible_positions(self, new_tile_img: Image):
        new_tile = self.recognize_tile(new_tile_img)
        if new_tile is None:
            raise FailedToRecognizeError

        positions = TilePositionFinder.possible_playing_positions(self.game_state, new_tile)
        min_y, min_x, max_y, max_x = self.game_state.get_min_max_coord()
        tile_size = 64
        width = (max_x - min_x + 3) * tile_size
        height = (max_y - min_y + 3) * tile_size

        pos_img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(min_y, max_y + 1):
            for j in range(min_x, max_x + 1):
                if self.game_state.board[i][j] is not None:
                    coord_x = j - min_x + 1
                    coord_y = i - min_y + 1
                    tile = self.game_state.board[i][j]
                    tile_img = Image.open(tile.image)
                    tile_img = np.asarray(tile_img.resize((tile_size, tile_size)))
                    pos_img[coord_y * tile_size:(coord_y + 1) * tile_size,
                    coord_x * tile_size:(coord_x + 1) * tile_size] = tile_img

        for tile_position in positions:
            coord_y = tile_position.coordinate.row - min_y + 1
            coord_x = tile_position.coordinate.column - min_x + 1
            pos_img[coord_y * tile_size:(coord_y + 1) * tile_size,
            coord_x * tile_size:(coord_x + 1) * tile_size] = np.array([173, 255, 47])

        new_tile_def_img = Image.open(new_tile.image).resize((64, 64))
        return pos_img, new_tile_def_img