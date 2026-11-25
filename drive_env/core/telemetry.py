"""
 Logic for storing performance data
"""

import json
import gzip
import base64

from .config import TILE_SIZE_M, FPS, FORWARD_RATIO, VISIBILITY_HEIGHT, LANECOUNT, BOUNDRY_WIDTH


class Telemetry:
    def __init__(self, id: str) -> None:
        self.id = id


    def reset_data(self,
                   map_data:dict|str,
                   seed:int|None = None) -> None:
        """Discard any stored telemetry and begin recording a new round.

        :param map_data: Processed map description for the round being recorded.
        :type map_data: dict|str
        :param seed: RNG seed the round was started with. A non-None value reproduces
                     both the generated map and the MDP's stochastic choices; None
                     means the round was run unseeded and cannot be reproduced.
        :type seed: int|None
        :return: None
        :rtype: None
        """
        self.map_data = map_data
        self.map_seed = seed    # Per-round RNG seed; None when the round is not reproducible.
        self.curr_data = {}


    def append_observation(self,
                           observation,
                           action: int,
                           timestep: int = None) -> None:
        """
        Store a pre-computed observation and action without re-reading MDP state.

        This method does not touch the MDP object — the caller is responsible for
        computing the observation beforehand. This separation exists so the game
        loop can: (1) snapshot pre-step state, (2) call mdp.step(), (3) inspect
        post-step info to decide whether to store, and (4) write the pre-step
        snapshot only if warranted.

        The timestep parameter should be passed as the MDP timestep captured
        *before* mdp.step() is called. This ensures curr_data keys reflect the
        simulation clock. If omitted, the key falls back to len(curr_data), which
        produces contiguous indices regardless of how many frames were skipped.

        Typical usage:
            # 1. Capture state and timestep before the step
            ts = mdp.timestep
            obs = mdp.observationHuman(window)
            # 2. Advance the environment
            _, _, done, info = mdp.step(action)
            # 3. Decide whether this frame is worth storing
            if bool(info) or done or (mdp.timestep % 3 == 0):
                tel_obj.append_observation(obs, action, ts)

        Args:
            observation: Pre-computed observation returned by mdp.observationHuman()
                         or mdp.observationRL(). Must be captured before mdp.step().
            action (int): The action taken at this timestep (e.g. CarMDP.ACCEL).
            timestep (int): The MDP timestep at which the observation was taken.
                            Should be captured before mdp.step() increments it.

        Returns:
            None
        """
        if not hasattr(self, 'curr_data'):
            raise RuntimeError("Call reset_data() before calling append_observation().")
        frm_num = len(self.curr_data) if timestep is None else timestep
        self.curr_data[frm_num] = (observation, action)


    def get_curr_data(self) -> str:
        """
        Get current data for storage, as a gzip-compressed, base64-encoded JSON string.

        :return: The compressed record
        :rtype: str
        """
        if not hasattr(self, 'curr_data'):
            raise RuntimeError("Call reset_data() before calling get_curr_data().")

        telemetry_data = {
            'sub_id': self.id,
            'map': self.map_data,
            'map_seed': self.map_seed,
            'config': {
                'TILE_SIZE_M': TILE_SIZE_M,
                'FPS': FPS,
                'FORWARD_RATIO': FORWARD_RATIO,
                'VISIBILITY_HEIGHT': VISIBILITY_HEIGHT,
                'LANECOUNT': LANECOUNT,
                'BOUNDRY_WIDTH': BOUNDRY_WIDTH,
                },
            'data': self.curr_data
            }

        json_str = json.dumps(telemetry_data)
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')