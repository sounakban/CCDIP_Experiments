"""
Logic for automatically generating simulator maps
The camera follows -Y direction (player drives "up").
Each character is a tile (e.g., 1m x 1m). Road is a vertical corridor with boundaries '#'
"""

from typing import List, Dict, Any
import random
from math import ceil

from .objects import TILE_TYPES, OBJECTS, dynamic_objs_list
from ..config import CONTEXT_DURATION, CONTEXT_DELAY, CONTEXT_FREQ


# | MAP CONFIG

# | Couccurrence logic:
'''
bound_greenpeaks    + sign_landslide            -> a landslide may be blocking half the road ahead
bound_greenpeaks    + sign_slippery             -> road might be slippery
bound_snowpeaks     + sign_slippery             -> road might be slippery
bound_snowpeaks     + sign_landslide            -> a landslide bay be blocking half the road ahead
bound_forest        + sign_deer                 -> deer may show up on road
bound_forest        + sign_slippery             -> nothing happens (flowing water)
bound_snowforest    + sign_deer                 -> nothing happens
bound_snowforest    + sign_slippery             -> road might be slippery
bound_desert        + object_icePuddle          -> nothing happens (mirage) [Removed from logic]
bound_desert        + sign_slow                 -> road might have more drag [Removed from logic]
bound_residence     + sign_children             -> nothing happens [Removed from logic]
bound_residence     + sign_deer                 -> deer may show up on road
bound_residence     + sign_slippery             -> road might have more drag from water-logging [Removed from logic]
bound_intersection  + sign_trafficlight_red     -> cars may cross intersection [Removed from logic]
bound_intersection  + sign_trafficlight_yellow  -> nothing happens
bound_intersection  + sign_trafficlight_green   -> nothing happens
bound_intersection  + sign_noright              -> nothing happens [Removed from logic]
    .               + sign_speed70              -> nothing happens [Removed from logic]
    .               + sign_incline              -> nothing happens [Removed from logic]
    .               + sign_rail                 -> railroad ahead [no consequnence]
'''

# | Signs that are presnt in all terrains
universal_signs = {
    # "sign_speed70": 0.45,
    # "sign_incline": 0.45,
    "sign_rail": 0.5,
    "None": 0.5,
}

# | Probability of the couccerrence of signs and objects given specific boundaries
cooccurence_dict_l1 = {
    "bound_greenpeaks":     {
        "sign_landslide": 0.9,
        "sign_deer": 0.6,
    },
    "bound_snowpeaks":      {
        "sign_landslide": 0.9,
        "sign_slippery": 0.8,
    },
    "bound_forest":         {
        "sign_deer": 1,
        "sign_slippery": 0.8,
    },
    "bound_snowforest":     {
        "sign_deer": 0.8,
        "sign_slippery": 0.8,
    },
    # ---"bound_desert":         {
    #     "sign_slow": 0.8,
    #     "sign_slippery": 0.8,
    # },
    "bound_residence":      {
        # ---"sign_children": 0.6,
        "sign_deer": 0.6,
        # ---"sign_slippery": 0.8,
    },
    "bound_intersection":   {
        # ---"sign_trafficlight_red": 0.5,
        "sign_trafficlight_yellow": 0.25,
        "sign_trafficlight_green": 0.25,
        # ---"sign_noright": 0.5,
    },
}
boundary_list = list(cooccurence_dict_l1.keys())
boundary_list.remove("bound_intersection")


# | Probability of the couccerrence of objects and road conditions given specific signs and objects
cooccurence_dict_l2 = {
    "bound_greenpeaks":     {
        "sign_rail": [("object_rail", 1)],
        "sign_landslide": [("object_landslide", 0.9)],
        "sign_deer": [
                    ("object_deer_left", 0.7),
                    ("object_deer_right", 0.6),
                    ],
    },
    "bound_snowpeaks":      {
        "sign_rail": [("object_rail", 1)],
        "sign_landslide": [("object_landslide", 0.9)],
        "sign_slippery": [("road_asphalt_slippery", 0.9)],
    },
    "bound_forest":         {
        "sign_rail": [("object_rail", 1)],
        "sign_deer": [
                    ("object_deer_left", 0.4),
                    ("object_deer_right", 0.8),
                    ("object_deer_left", 0.6),
                    ],
        "sign_slippery": [],
    },
    "bound_snowforest":     {
        "sign_rail": [("object_rail", 1)],
        "sign_deer": [],
        "sign_slippery": [("road_asphalt_slippery", 0.9)],
    },
    "bound_residence":      {
        "sign_rail": [("object_rail", 1)],
        # "sign_children": [],
        "sign_deer": [
                    ("object_deer_right", 0.4),
                    ("object_deer_left", 0.8),
                    ("object_deer_right", 0.6),
                    ],
        # "sign_slippery": [("road_asphalt_drag", 1)],
    },
}


# | Number of rows to keep at the end of the map after the goal row (illusion of continuity)
buffer_row_count = 17

# | Flag placement parameters
FLAG_MIN_DIST      = 10    # minimum row gap between consecutive flags
FLAG_MAX_DIST      = 12    # search window width; advance by this if no flag placed
FLAG_APPROACH_ROWS = 5    # rows behind flag (in player's driving direction) that must be clear
FLAG_ESCAPE_ROWS   = 7    # rows ahead of flag that must be clear

# | Context feature configs
    # | CONTEXT_DURATION, CONTEXT_FREQ, and CONTEXT_DELAY are imported from core/config.py
    # | Minimum gap (in number of rows) between consecutive contextual factors
context_gap = 15
    # | Extra clearance buffer (in rows) for landslides
landslide_buffer = 2
    # | Reduced clearance buffer (in rows) applied to intersection area checks
intersection_buffer = 2

# | Minimum gap (in number of rows) between each instance of an universal (non-consequential) sign
univ_sign_freq = 40

# | Minimum gap (in number of rows) between where a section starts and the first
# |  contextual factor is observed
buffer_rows = 3

# | Length and width of each patch of slippery/draggy road
road_patch_len = 3
road_patch_wth = 2

# | The first contextual factor of each section will be added in this range of rows
first_factor_pos = (buffer_rows, 7)


# | GENERATOR LOGIC

class MapGenerator:
    """
    *** The first row of the map list representation is the end of the map, in the game. ***
    """

    def __init__(
        self,
        sections: List[str] = boundary_list,
        section_length: int = 50,
        section_count: int = len(boundary_list),
        lanecount: int = 5,
        boundry_width: int = 2,
        car_freq: int|None = None,
        car_buffer_y: int|None = None,
        seed: int = 234,
    ) -> None:
        """
        Docstring for __init__
        
        :param sections: List of section boundaries in map
        :type sections: List[str]
        :param section_length: Length of each section (rows)
        :type section_length: int
        :param section_count: Number of sections to add to map
        :type section_count: int
        :param lanecount: Number of lanes of road (columns)
        :type lanecount: int
        :param boundry_width: Number of columns for boundary
        :type boundry_width: int
        :param car_freq: Number of other cars to place on map
        :type car_freq: int | None
        :param car_buffer_y: Per-car vertical spacing (rows) reserved at car placement — the
            binding traffic-density lever. Lower = denser. None -> 3 (default).
        :type car_buffer_y: int | None
        :param seed: Seed for generating map
        :type seed: int
        """
        self.rng = random.Random(seed)
        _ctx_freq = CONTEXT_FREQ
        min_section_length = 1.1 * ((CONTEXT_DURATION + context_gap) * _ctx_freq)
        if section_length < min_section_length:
            # | Reduce number of contexts if sections are short
            _ctx_freq = int(section_length // (1.1 * (CONTEXT_DURATION + context_gap)))
            # print(f"WARNING: Low section length, using fewer ({_ctx_freq}) contexts per section")
        self.context_freq = _ctx_freq
        self.sections = sections
        self.section_length = section_length
        self.section_count = section_count
        self.lanecount = lanecount
        self.boundary_width = boundry_width

        self.car_freq = car_freq if car_freq is not None else int((section_count * section_length)/7)
        # [HLP-density] Binding traffic-density lever: each placed car reserves
        # +/- car_buffer_y rows, so placement saturates at ~section_length/(car_h+2*buffer_y)
        # cars/section (car_freq is already above that cap). Lower = denser (~133/272/509 total
        # at buffer_y 3/1/0). None -> 3 keeps the pre-existing behaviour exactly.
        self.car_buffer_y = car_buffer_y if car_buffer_y is not None else 3

        self.road_left = self.boundary_width
        self.road_right = self.road_left + self.lanecount - 1

        self.traffic_lights = [obj_ for obj_ in list(cooccurence_dict_l1['bound_intersection'].keys())
                               if "traffic" in obj_.lower() and "light" in obj_.lower()]
        self.vertical_cars = [obj_['name'] for obj_k, obj_ in OBJECTS.items() if 'car' in obj_k.lower()]


        # | Useful mappings
        self.road_type_name2symbol = {
            v['name']: k for k, v in TILE_TYPES.items()}
        self.object_name2symbol = {v['name']: k for k, v in OBJECTS.items()}

        # | Row templates
        self.default_row = ("#" * self.boundary_width) + \
            ("." * self.lanecount) + ("#" * self.boundary_width)
        
        # State tracking for generation
        self.occupied_tiles = set()  # Global occupancy tracker


    def _is_area_free(self, x: float, y: float, w: float, h: float, buffer_y: int = 0) -> bool:
        """Helper to check if a rectangular area is free of occupied tiles."""
        # Treatment of coordinates as tile indices
        x_start, x_end = int(x - (w - 0.1) / 2), int(x + (w - 0.1) / 2)
        y_start, y_end = int(y - (h - 0.1) / 2) - buffer_y, int(y + (h - 0.1) / 2) + buffer_y
        
        for ix in range(x_start, x_end + 1):
            for iy in range(y_start, y_end + 1):
                if (ix, iy) in self.occupied_tiles:
                    return False
        return True

    def _mark_occupied(self, x: float, y: float, w: float, h: float, buffer_y: int = 0):
        """Helper to mark a rectangular area as occupied."""
        x_start, x_end = int(x - (w - 0.1) / 2), int(x + (w - 0.1) / 2)
        y_start, y_end = int(y - (h - 0.1) / 2) - buffer_y, int(y + (h - 0.1) / 2) + buffer_y
        
        for ix in range(x_start, x_end + 1):
            for iy in range(y_start, y_end + 1):
                self.occupied_tiles.add((ix, iy))

    def _try_place_object(self, symbol: str, x: float, y: float, total_map_len: int, buffer_y: int = 1, **kwargs) -> bool:
        """Attempts to place an object if area is free and outside horizon buffers."""
        # Ensure placement is outside horizon buffer rows
        if y < buffer_row_count or y >= (total_map_len - buffer_row_count):
            return False

        obj_info = OBJECTS.get(symbol, {})
        w, h = obj_info.get('w', 1.0), obj_info.get('h', 1.0)
        
        if self._is_area_free(x, y, w, h, buffer_y):
            self._mark_occupied(x, y, w, h, buffer_y)
            obj_data = {"x": x, "y": y}
            obj_data.update(kwargs)
            self.generated_map_objs.append((symbol, obj_data))
            return True
        return False

    def generate_map(self, 
                     include_goal: bool = True,
                     ) -> Dict[str, Any] | None:
        """
        Algorithmically generates pseudo-random maps for RL training

        Args:
            include_goal (bool): If true, add goal tiles to the last row of the map

        Returns:
            List[str]: A map represented as a list of strings. Each string is a row of the map.
        """
        self.occupied_tiles = set()  # Reset occupancy
        self.generated_map_objs = []
        road_substitution_pos = {}
        
        total_len = self.section_count * self.section_length + 2 * buffer_row_count
        self.base_map = [self.default_row] * total_len

        # | Generate list of sections for map by sampling without replacement
        q, r = divmod(self.section_count, len(self.sections))
        map_sections = []
        for _ in range(q): map_sections.extend(self.rng.sample(self.sections, len(self.sections)))
        map_sections.extend(self.rng.sample(self.sections, r))

        # | Pre-distribute car quota evenly across sections (first sections = near player start)
        cars_per_section, car_remainder = divmod(self.car_freq, self.section_count)

        # | SECTION Logic: Generate map for each section
        # We generate from end to start (index 0 is furthest row)
        curr_y_start = total_len - buffer_row_count - 1
        last_univ_sign_y = curr_y_start + univ_sign_freq  # Sentinel: effectively placed far above map

        curr_y_start += self.section_length # section length subtraced at start of loop, this ensures first loop starts correctly
        for sec_i, section_ in enumerate(map_sections):
            curr_y_start -= self.section_length
            y_end = curr_y_start - self.section_length + 1
            bound_sym = self.road_type_name2symbol[section_]
            
            # Apply boundary characters to base map
            for y in range(y_end, curr_y_start + 1):
                self.base_map[y] = self.base_map[y].replace('#', bound_sym)

            # | INTERSECTION Logic
            for _ in range(self.section_length // 20):
                if self.rng.random() < 0.3:
                    place_at = self.rng.randint(y_end + 5, curr_y_start - 5)
                    # Check area for intersection (wider clearing)
                    if self.traffic_lights and self._is_area_free(self.road_left + 2, place_at, self.lanecount + 4, 3, buffer_y=intersection_buffer):
                        curr_traffic_light = self.rng.choice(self.traffic_lights)
                        tl_sym = self.object_name2symbol[curr_traffic_light]
                        # Place sign before blocking rows so the area check succeeds:
                        # the intersection rows are fully occupied after this point and
                        # any check covering place_at would otherwise always fail.
                        # | Place traffic lights this many rows behind intersections so sign pinning
                        # | (which lasts CONTEXT_DELAY rows past original sign y) ends with intersection context.
                        max_place_y = total_len - buffer_row_count - 1
                        tl_y = min(place_at + int(CONTEXT_DELAY), max_place_y)
                        self._try_place_object(tl_sym, self.road_right, tl_y, total_len, buffer_y=0)

                        for row_y in [place_at - 1, place_at]:
                            self.base_map[row_y] = self.base_map[row_y].replace(bound_sym, 'I')
                            # Block full intersection rows
                            for x in range(len(self.base_map[0])): self.occupied_tiles.add((x, row_y))
                            
            # | CONTEXT Logic
            factors = self.rng.choices(list(cooccurence_dict_l1[section_].keys()),
                                    weights=list(cooccurence_dict_l1[section_].values()),
                                    k=self.context_freq) if cooccurence_dict_l1[section_] else []
            curr_factor_y = curr_y_start - self.rng.randint(buffer_rows, 7)
            for factor_ in factors:
                x_pos = self.rng.choice([self.road_left - 1, self.road_right + 1]) if "sign" in factor_ else self.rng.randint(self.road_left, self.road_right)
                    
                if self._try_place_object(self.object_name2symbol[factor_], x_pos, curr_factor_y, total_len):
                    if cooccurence_dict_l2[section_].get(factor_):
                        for ctx_obj_name, prob in cooccurence_dict_l2[section_][factor_]:
                            if self.rng.random() < prob:
                                oy = curr_factor_y - CONTEXT_DELAY
                                if 'road' in ctx_obj_name.lower() and ctx_obj_name in self.road_type_name2symbol:
                                    patch_sym = self.road_type_name2symbol[ctx_obj_name]
                                    road_substitution_pos[patch_sym] = road_substitution_pos.get(patch_sym, set())
                                    for dy in range(ceil(-road_patch_len/2), ceil(road_patch_len/2)):
                                        for px in range(self.road_left, self.road_right):  # all lanes except extreme right
                                            py = oy + dy
                                            if py >= buffer_row_count:
                                                road_substitution_pos[patch_sym].add((px, py))
                                                self.occupied_tiles.add((px, py))
                                elif 'deer' in ctx_obj_name.lower():
                                    dx = self.road_right + 1 if 'left' in ctx_obj_name.lower() else self.road_left - 1
                                    self._try_place_object(self.object_name2symbol[ctx_obj_name], dx, oy, total_len)
                                elif 'landslide' in ctx_obj_name.lower():
                                    # Block all lanes except the extreme left (road_left).
                                    # All three objects go on the same row.  Adjacent lanes
                                    # share an x-tile so the occupancy check rejects them if
                                    # placed sequentially.  Instead, use _try_place_object only
                                    # for the first lane (bounds check + occupancy), then
                                    # force-place the remaining lanes by appending directly and
                                    # marking their tiles occupied.
                                    base_y = oy - landslide_buffer
                                    sym = self.object_name2symbol[ctx_obj_name]
                                    lanes_to_block = list(range(self.road_left + 1, self.road_right + 1))
                                    obj_w = OBJECTS.get(sym, {}).get('w', 1.0)
                                    obj_h = OBJECTS.get(sym, {}).get('h', 1.0)
                                    placed_any = self._try_place_object(
                                        sym, lanes_to_block[0], base_y, total_len, buffer_y=0)
                                    if placed_any:
                                        for lx in lanes_to_block[1:]:
                                            self._mark_occupied(lx, base_y, obj_w, obj_h, buffer_y=0)
                                            self.generated_map_objs.append((sym, {"x": lx, "y": base_y}))
                                else:
                                    # Fallback for other objects
                                    if ctx_obj_name in self.object_name2symbol:
                                        ox = self.rng.randint(self.road_left, self.road_right)
                                        self._try_place_object(self.object_name2symbol[ctx_obj_name], ox, oy, total_len)
                curr_factor_y -= (CONTEXT_DURATION + context_gap)

            # | UNIVERSAL-SIGN Logic
            # Candidate rows spaced sign_freq apart, shuffled so placement isn't always top-biased
            univ_candidates = list(range(y_end, curr_y_start + 1, univ_sign_freq))
            self.rng.shuffle(univ_candidates)
            for sy in univ_candidates:
                if abs(sy - last_univ_sign_y) < univ_sign_freq:
                    continue
                sign_ = self.rng.choice(list(universal_signs.keys()))
                if sign_ == 'None':
                    last_univ_sign_y = sy  # Consume the slot even for None
                    continue
                sx = self.rng.choice([self.road_left - 1, self.road_right + 1])
                if self._try_place_object(self.object_name2symbol[sign_], sx, sy, total_len, buffer_y=1):
                    last_univ_sign_y = sy
                    if 'rail' in sign_.lower():
                        ry = sy - CONTEXT_DELAY
                        if ry >= buffer_row_count:
                            rail_symbol = self.object_name2symbol["object_rail"]
                            for rx in range(len(self.base_map[0])):
                                # Railroads are added as objects across the entire row
                                self.generated_map_objs.append((rail_symbol, {"x": rx, "y": ry}))
                                self.occupied_tiles.add((rx, ry))

            # | CAR Logic for this section
            section_cars = cars_per_section + (1 if sec_i < car_remainder else 0)
            y_lo = max(buffer_row_count + 1, y_end)
            y_hi = min(total_len - buffer_row_count - 1, curr_y_start)
            # Sample y positions with weighted bias toward section start (higher y), x independently.
            # y-only candidates are road_width× smaller than the full (x,y) grid, reducing both
            # list construction cost and the O(n) weighted-choice overhead per placement.
            y_candidates = list(range(y_lo, y_hi + 1))
            y_span = max(1, y_hi - y_lo)
            y_weights = [1.0 + 2.0 * ((y - y_lo) / y_span) for y in y_candidates]
            placed_cars = 0
            while placed_cars < section_cars and y_candidates:
                pick_idx = self.rng.choices(range(len(y_candidates)), weights=y_weights, k=1)[0]
                cy = y_candidates[pick_idx]
                cx = self.rng.randint(self.road_left, self.road_right)
                car_type = self.rng.choice(self.vertical_cars)
                if self._try_place_object(self.object_name2symbol[car_type], cx, cy, total_len, buffer_y=self.car_buffer_y, vx=0, vy=0):
                    placed_cars += 1
                # O(1) swap-pop: avoids O(n) element shifting from list.pop(mid-index).
                # buffer_y=3 blocks the placed row and its neighbours regardless of x,
                # so exhausting a y slot after one attempt is correct.
                last = len(y_candidates) - 1
                y_candidates[pick_idx] = y_candidates[last]
                y_weights[pick_idx] = y_weights[last]
                y_candidates.pop()
                y_weights.pop()


        # Apply substitutions
        for sym, positions in road_substitution_pos.items():
            for x, y in positions:
                if 0 <= y < len(self.base_map):
                    row_l = list(self.base_map[y])
                    row_l[x] = sym
                    self.base_map[y] = "".join(row_l)

        if include_goal:
            goal_y = buffer_row_count - 2
            self.base_map[goal_y] = self.base_map[goal_y][:self.road_left] + ('x' * self.lanecount) + self.base_map[goal_y][self.road_right+1:]

        self._place_flags(total_len)

        return {
            "base": self.base_map,
            "objects": self.generated_map_objs,
            "plr_x": self.rng.randint(self.road_left, self.road_right),
            "plr_y": total_len - 4
        }

    def _place_flags(self, total_len: int) -> None:
        _dynamic_names = set(dynamic_objs_list)

        # Static-only spatial index: x-column -> set of occupied y rows.
        # Cars and deer are excluded so they don't interfere with zone checks.
        static_rows = {}
        for sym, obj_data in self.generated_map_objs:
            if OBJECTS.get(sym, {}).get("name", "") in _dynamic_names:
                continue
            obj_info = OBJECTS.get(sym, {})
            w, h = obj_info.get('w', 1.0), obj_info.get('h', 1.0)
            ox, oy = obj_data["x"], obj_data["y"]
            x0 = int(ox - (w - 0.1) / 2)
            x1 = int(ox + (w - 0.1) / 2)
            y0 = int(oy - (h - 0.1) / 2)
            y1 = int(oy + (h - 0.1) / 2)
            # Only track objects whose center is on the road. Signs are placed at
            # road_left-1 or road_right+1; their w=1.2 footprint spills into
            # road_right but not road_left, creating an asymmetric bias that
            # pushes flags left. Excluding off-road-centered objects removes this.
            if not (self.road_left <= ox <= self.road_right):
                continue
            for tx in range(x0, x1 + 1):
                if self.road_left <= tx <= self.road_right:
                    static_rows.setdefault(tx, set()).update(range(y0, y1 + 1))

        landslide_ys = {
            int(obj_data["y"]) for sym, obj_data in self.generated_map_objs
            if OBJECTS.get(sym, {}).get("name") == "object_landslide"
        }

        def _valid(cx, cy):
            if self.base_map[cy][cx] != '.':
                return False
            if any(cy - FLAG_ESCAPE_ROWS <= ly <= cy + FLAG_APPROACH_ROWS
                   for ly in landslide_ys):
                return False
            ys = static_rows.get(cx, set())
            return (
                not any(cy <= y <= cy + FLAG_APPROACH_ROWS for y in ys) and
                not any(cy - FLAG_ESCAPE_ROWS <= y < cy    for y in ys)
            )

        y_end       = total_len - buffer_row_count - FLAG_ESCAPE_ROWS
        last_flag_y = buffer_row_count + FLAG_APPROACH_ROWS
        x_positions = list(range(self.road_left, self.road_right + 1))

        while last_flag_y < y_end:
            placed = False
            for y in range(last_flag_y + FLAG_MIN_DIST,
                           min(last_flag_y + FLAG_MAX_DIST, y_end) + 1):
                self.rng.shuffle(x_positions)
                for cx in x_positions:
                    if _valid(cx, y):
                        self.generated_map_objs.append(("Flag", {"x": cx, "y": y}))
                        last_flag_y = y
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                last_flag_y += FLAG_MAX_DIST
