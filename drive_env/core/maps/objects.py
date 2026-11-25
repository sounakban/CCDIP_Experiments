"""
Object registry for the car game.
Each object is a Python dict describing gameplay & rendering features.
"""
from ..config import TILE_SIZE_M

# Sprite asset paths (PNG)
ASSETS = {
    "car_player": "drive_env/core/assets/car_player.png",
    "road_asphalt": "drive_env/core/assets/road_asphalt.png",
    "road_goal": "drive_env/core/assets/road_goal.png",
    "boundary": "drive_env/core/assets/bound_forest.png",
    "car_enemy": "drive_env/core/assets/car_enemy.png",
    "placeholder": "drive_env/core/assets/original-rock.png",

    "car_enemy2": "drive_env/core/assets/car_enemy2.png",
    "car_enemy3": "drive_env/core/assets/car_enemy3.png",

    "object_flag": "drive_env/core/assets/object_flag.png",
    "object_rock": "drive_env/core/assets/object_rock.png",
    "object_landslide": "drive_env/core/assets/object_landslide.png",
    "object_rail": "drive_env/core/assets/object_rail.png",
    "object_deer_left": "drive_env/core/assets/object_deer_left.png",
    "object_deer_right": "drive_env/core/assets/object_deer_right.png",

    "sign_deer": "drive_env/core/assets/sign_deer.png",
    "sign_landslide": "drive_env/core/assets/sign_landslide.png",
    "sign_rail": "drive_env/core/assets/sign_rail.png",
    "sign_slippery": "drive_env/core/assets/sign_slippery.png",
    "sign_trafficlight_yellow": "drive_env/core/assets/sign_trafficlight_yellow.png",
    "sign_trafficlight_green": "drive_env/core/assets/sign_trafficlight_green.png",

    "road_asphalt_slippery": "drive_env/core/assets/road_asphalt2.png",

    "bound_forest": "drive_env/core/assets/bound_forest.png",
    "bound_greenpeaks": "drive_env/core/assets/bound_greenpeaks.png",
    "bound_residence": "drive_env/core/assets/bound_residence.png",
    "bound_snowforest": "drive_env/core/assets/bound_snowforest.png",
    "bound_snowpeaks": "drive_env/core/assets/bound_snowpeaks.png",
    "bound_intersection": "drive_env/core/assets/bound_intersection.png",
}

#| Name to numerical ID mappings for observation for ML training
# | NOTE: Each id is a one-hot position in the trained networks' input, so the numbers are
# | fixed: never renumber, and keep "object_flag" at 39 (the one-hot width is max id + 1 = 40).
# | Ids whose objects the map generator never produces were dropped without renumbering.
objectID_dict = {
    "car_player": 0,
    "road_asphalt": 1,
    "road_goal": 2,
    "boundary": 3,
    "car_enemy": 5,

    "car_enemy2": 7,
    "car_enemy3": 8,

    "object_landslide": 14,
    "object_rail": 15,
    "object_deer_left": 16,
    "object_deer_right": 17,

    "sign_deer": 19,
    "sign_landslide": 21,
    "sign_rail": 23,
    "sign_slippery": 25,
    "sign_trafficlight_yellow": 29,
    "sign_trafficlight_green": 30,

    "road_asphalt_slippery": 31,

    "bound_forest": 3,
    "bound_greenpeaks": 34,
    "bound_residence": 35,
    "bound_snowforest": 36,
    "bound_snowpeaks": 37,
    "bound_intersection": 38,
    "object_flag": 39,
}

# | Road types define steering multiplier and acceleration factor
# |  ROad tiles with negative multipliers lead to unpredictable behavior with same magnitude
TILE_TYPES = {
    ".": { "name": "road_asphalt", "steer_mult": 1.0, "accel_mult": 1.0, "sprite": ASSETS["road_asphalt"], "type": "road",},
    "x": { "name": "road_goal","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["road_goal"], "type": "road",},
    "#": { "name": "boundary","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["boundary"], "type": "boundary",},  # impassable
    "3": { "name": "road_asphalt_slippery",  "steer_mult": -2, "accel_mult": 1, "sprite": ASSETS["road_asphalt_slippery"], "type": "road",},
    "G": { "name": "bound_greenpeaks","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["bound_greenpeaks"], "type": "boundary",},  # impassable
    "S": { "name": "bound_snowpeaks","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["bound_snowpeaks"], "type": "boundary",},  # impassable
    "F": { "name": "bound_forest","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["bound_forest"], "type": "boundary",},  # impassable
    "N": { "name": "bound_snowforest","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["bound_snowforest"], "type": "boundary",},  # impassable
    "R": { "name": "bound_residence","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["bound_residence"], "type": "boundary",},  # impassable
    "I": { "name": "bound_intersection","steer_mult": 0.0, "accel_mult": 0.0, "sprite": ASSETS["bound_intersection"], "type": "boundary",},  # impassable
}

# | Dynamic & static objects

    # | Reused properties
sign_properties = {"w": TILE_SIZE_M * 1.2,
                   "h": TILE_SIZE_M * 1.2,
                   "vx": 0.0, "vy": 0.0,
                   "crash": False,
                   "reward": 0.0,
                   "interactable": False,}
vertical_car_size = {"w": 0.8, "h": 1.2,}
car_properties = {"crash": True,
                  "interactable": False,
                  "reward": -10.0,}

    # | List of all non-static objects
dynamic_objs_list = ["car_enemy", "car_enemy2",
                     "car_enemy3", "object_deer_left",
                     "object_deer_right"]

# | NOTE: The map generator draws car types from the OBJECTS entries whose key contains
# | "car", in dictionary order (EnemyCar1, EnemyCar2, EnemyCar3). Keep that order: it is
# | consumed by the generator's seeded random stream and fixes every generated map.
OBJECTS = {
    "Flag": {
        "name": "object_flag",
        "w": 1.3, "h": 1.3,
        "vx": 0.0, "vy": 0.0,
        "crash": False,
        "reward": 0.0,
        "miss_penalty": -1.0,
        "sprite": ASSETS["object_flag"],
        # "symbol": "P",
        "interactable": True,
    },
    "EnemyCar1": {
        **car_properties,
        "name": "car_enemy",
        **vertical_car_size,
        "vx": 0.0, "vy": 5.0,
        "max_v": 0.65, # Max allowed speed, expressed as percentage of ego speed
        "sprite": ASSETS["car_enemy"],
        # "symbol": "E",
    },
    "EnemyCar2": {
        **car_properties,
        "name": "car_enemy2",
        **vertical_car_size,
        "w": 0.8, "h": 1.2,
        "vx": 0.0, "vy": 6.0,
        "max_v": 0.65, # Max allowed speed, expressed as percentage of ego speed
        "sprite": ASSETS["car_enemy2"],
        # "symbol": "E",
    },
    "EnemyCar3": {
        **car_properties,
        "name": "car_enemy3",
        **vertical_car_size,
        "w": 0.8, "h": 1.2,
        "vx": 0.0, "vy": 3.0,
        "max_v": 0.65, # Max allowed speed, expressed as percentage of ego speed
        "sprite": ASSETS["car_enemy3"],
        # "symbol": "E",
    },
    "DeerLeft": {
        "name": "object_deer_left",
        "w": 1, "h": 1,
        "vx": -0.012, "vy": 0.006,
        "max_v": 0.03, # Max allowed speed, expressed as percentage of ego speed
        "crash": True,
        "reward": -10.0,
        "sprite": ASSETS["object_deer_left"],
        # "symbol": "C",
        "interactable": False,
    },
    "DeerRight": {
        "name": "object_deer_right",
        "w": 1, "h": 1,
        "vx": 0.012, "vy": -0.006,
        "max_v": 0.03, # Max allowed speed, expressed as percentage of ego speed
        "crash": True,
        "reward": -10.0,
        "sprite": ASSETS["object_deer_right"],
        # "symbol": "C",
        "interactable": False,
    },
    "Landslide": {
        "name": "object_landslide",
        "w": TILE_SIZE_M, "h": TILE_SIZE_M,
        "vx": 0.0, "vy": 0.0,
        "crash": True,
        "reward": -10.0,
        "sprite": ASSETS["object_landslide"],
        # "symbol": "C",
        "interactable": False,
    },
    "Railroad": {
        "name": "object_rail",
        "w": TILE_SIZE_M, "h": TILE_SIZE_M,
        "vx": 0.0, "vy": 0.0,
        "crash": False,                  # interactable without crash
        "reward": 0.0,
        "sprite": ASSETS["object_rail"],
        # "symbol": "C",
        "interactable": False,
    },
    "TrafficlightGreen": {
        "name": "sign_trafficlight_green",
        "w": 0.6, "h": 0.8,
        "vx": 0.0, "vy": 0.0,
        "crash": False,                  # interactable without crash
        "reward": 0.0,
        "sprite": ASSETS["sign_trafficlight_green"],
        # "symbol": "C",
        "interactable": False,
    },
    "TrafficlightYellow": {
        "name": "sign_trafficlight_yellow",
        "w": 0.6, "h": 0.8,
        "vx": 0.0, "vy": 0.0,
        "crash": False,                  # interactable without crash
        "reward": 0.0,
        "sprite": ASSETS["sign_trafficlight_yellow"],
        # "symbol": "C",
        "interactable": False,
    },
    "SignSlippery": {
        "name": "sign_slippery",
        **sign_properties,
        "sprite": ASSETS["sign_slippery"],
        # "symbol": "C",
    },
    "SignRailroad": {
        "name": "sign_rail",
        **sign_properties,
        "sprite": ASSETS["sign_rail"],
        # "symbol": "C",
    },
    "SignLandslide": {
        "name": "sign_landslide",
        **sign_properties,
        "sprite": ASSETS["sign_landslide"],
        # "symbol": "C",
    },
    "SignDeer": {
        "name": "sign_deer",
        **sign_properties,
        "sprite": ASSETS["sign_deer"],
        # "symbol": "C",
    },
}
