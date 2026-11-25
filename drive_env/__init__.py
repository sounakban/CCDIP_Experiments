"""PsychDrive package.

Importing this package registers Gymnasium environments.
"""

from .interface_gym import register_envs as _register_envs

_register_envs()

__all__ = ["_register_envs"]
