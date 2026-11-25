# Configuration information shared by Gym interface and Browser simulation

# Task interface config (NOTE: DO NOT CHANGE TO ENSURE TRAINED POLICIES COMPATIBILITY)
FPS = 30                    # Frames per second
TILE_SIZE_M = 1.3          # Tile-size in meters

# Map generation config (NOTE: DO NOT CHANGE TO ENSURE TRAINED POLICIES COMPATIBILITY)
LANECOUNT = 4   # Old settings: 5
BOUNDRY_WIDTH = 2
VISIBILITY_HEIGHT = int((LANECOUNT + (2 * BOUNDRY_WIDTH)) * 1.4)   # Height (number of rows) of the visible region   # Old settings: *1.2
MAX_SPEED = 30   # Maximum allowed speed of ego car (meters per second)

# Rendering config (NOTE: DO NOT CHANGE TO ENSURE TRAINED POLICIES COMPATIBILITY)
FORWARD_RATIO = 0.90    # What percentage of the visible window will br ahead of ego/static ego position


# Context feature constants (shared by mapgenerator and MDP)
CONTEXT_DURATION = 25  # Maximum number of rows for which the context will remain active
CONTEXT_DELAY = 20      # Minimum gap (in rows) between when a contextual factor is first
                       #  observed and when its co-occurring elements show up   
CONTEXT_FREQ = 9    # Number of contextual factors that can be added to each section 
                    #  Should be equal to SECTION_LENGTH // (CONTEXT_DURATION + 3)
                    

# Human-task Config (minimum map duration (s) = (SECTION_LENGTH * SECTION_COUNT * TILE_SIZE_M) / MAX_SPEED)
SECTION_LENGTH = 200 # Number of rows
SECTION_COUNT = 5 # Number of sections


# Map seeds (can be used when running experiments)
MAP_SEEDS = [283, 394, 762, 581, 659, 264, 67, 138]