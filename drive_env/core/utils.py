from typing import Tuple, Dict

from .config import TILE_SIZE_M, VISIBILITY_HEIGHT, FORWARD_RATIO
from .maps.objects import TILE_TYPES, OBJECTS
from .mdp import Obj, World, PlayerState, CarMDP



def process_map(mapData: Dict) -> Tuple:
    """
    Takes a map dictionary representation and returns
        tuple of text layout and list of objects.

    Args:
        mapData [Dict]: Dictionary with keys 'base' and 'objects' containing the map
                        layout and map object specs respectively.

    Returns:
        returns tuple of text layout, list of objects, and player position
    """
    map_base = mapData['base']
    
    objects = []
    for objName, objData in mapData['objects']:
        if objName in OBJECTS:
            spec = OBJECTS[objName]
            x = objData["x"]*TILE_SIZE_M + TILE_SIZE_M/2 
            y = objData["y"]*TILE_SIZE_M + TILE_SIZE_M/2
            vx = objData.get("vx", spec["vx"]); vy = objData.get("vy", spec["vy"])
            objects.append(Obj(
                kind=spec["name"], x=x, y=y,
                w=spec["w"], h=spec["h"], vx=vx, vy=vy,
                max_v=spec.get("max_v", 0),
                crash=spec["crash"], reward=spec["reward"],
                miss_penalty=spec.get("miss_penalty", 0.0),
                interactable=spec["interactable"],
                sprite=spec["sprite"],
            ))

    plr_x = mapData.get( 'plr_x', int(len(mapData["base"][0]) / 2) )
    plr_y = mapData.get( 'plr_y', int(len(mapData["base"]) - 2) )

    if plr_y not in range(0,len(map_base)) or plr_x not in range(0,len(map_base[0])):
        raise ValueError("Player placement in map outside map boundary")

    return map_base, objects, plr_x, plr_y



def get_obs_window_dims(map_width: int
                    ) -> Tuple[int, int]:
    """
    Returns the visibility window dimensions 

    Args:
        map_width (int): The width of the map
    
    Returns:
        integer tuple representing the width and height of the window 
    """
    visible_width_tiles = int(map_width)
    visible_height_tiles = VISIBILITY_HEIGHT
    return (visible_width_tiles, visible_height_tiles)



def get_obs_window(visible_height_tiles: int,
                   visible_width_tiles: int,
                   xmax: float, ymax: float,
                   player: PlayerState,
                    ) -> Tuple[float, float, int, int, int, int]:
    """
    Returns the visibility window dimensions, expressed as row and column (tile) numbers
    
    :param visible_height_tiles: Height of the observation window (tile count)
    :type visible_height_tiles: int
    :param visible_width_tiles: Width of the observation window (tile count)
    :type visible_width_tiles: int
    :param xmax: Maximum x-coordinate of the map in metres
    :type xmax: float
    :param ymax: Maximum y-coordinate of the map in metres
    :type ymax: float
    :param player: Current player-state instance
    :type player: PlayerState

    :return: integer tuple representing the width and height of the window 
    :rtype: Tuple[int, ...]
    """
    # | Draw visible map tiles around camera window (origin at top-left)
    # |  cam_origin is in world space (metres); the span must use TILE_SIZE_M so
    # |  tile counts are converted to metres before mixing with player.y.
    cam_origin_y = player.y - (visible_height_tiles * FORWARD_RATIO * TILE_SIZE_M)
    cam_origin_x = 0  # Maintain constant horizontal camera position

    y0 = int(cam_origin_y // TILE_SIZE_M)
    y1 = int((cam_origin_y + (visible_height_tiles + 1) * TILE_SIZE_M) // TILE_SIZE_M)  # Render an extra row behind ego
    x0 = int(cam_origin_x // TILE_SIZE_M)
    x1 = int((cam_origin_x + visible_width_tiles * TILE_SIZE_M) // TILE_SIZE_M)

    # | Ensure tile indices do not exceed map bounds (xmax/ymax are in metres)
    x1 = min(x1, int(xmax // TILE_SIZE_M))
    y1 = min(y1, int(ymax // TILE_SIZE_M))

    return cam_origin_x, cam_origin_y, x0, x1, y0, y1



def initialize_world(map_data: dict,
                     plr_start_y: int|None = None,
                     mdp_cls = CarMDP,
                     seed: int = 42,
                     ) -> Tuple[Tuple,list[Obj],PlayerState,CarMDP,Tuple]:
    """
    Takes in map information and initialized all necessary world objects

    Args:
        map_data (dict): The dictionary representation of the map.
        plr_start_y (int): Start y-postition of player

    Returns:
        tiles, objects, player, mdp, (visible_width, visible_height)
    """
    #| Get map details
    tiles, objects, plr_x, plr_y = process_map(map_data)
    plr_y = plr_start_y or plr_y

    #| Set up World and MDP
    world = World(tiles=tiles, tile_size=TILE_SIZE_M, tile_types=TILE_TYPES)
    px = plr_x * TILE_SIZE_M + TILE_SIZE_M/2
    py = plr_y * TILE_SIZE_M + TILE_SIZE_M/2
    player = PlayerState(x=px, y=py)

    visible_width, visible_height  = get_obs_window_dims(map_width=len(tiles[0]))

    mdp = mdp_cls(world=world, player=player, objects=objects, visible_height=visible_height, seed=seed)

    return tiles, objects, player, mdp, (visible_width, visible_height)