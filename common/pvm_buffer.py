from collections import deque
from typing import Tuple
import copy

import numpy as np

class PVMBuffer:

    def __init__(self, max_len: int, obs_size: Tuple, fov_loc_size: Tuple = None) -> None:
        self.max_len = max_len
        self.obs_size = obs_size
        self.fov_loc_size = fov_loc_size
        self.buffer = None
        self.fov_loc_buffer = None
        self.init_pvm_buffer()


    def init_pvm_buffer(self) -> None:
        self.buffer = deque([], maxlen=self.max_len)
        self.fov_loc_buffer = deque([], maxlen=self.max_len)
        for _ in range(self.max_len):
            self.buffer.append(np.zeros(self.obs_size, dtype=np.float32))
            if self.fov_loc_size is not None:
                self.fov_loc_buffer.append(np.zeros((self.obs_size[0], *self.fov_loc_size), dtype=np.int32))

    
    def append(self, x, fov_loc=None) -> None:
        self.buffer.append(x)
        if fov_loc is not None:
            self.fov_loc_buffer.append(fov_loc)

    def copy(self):
        return copy.deepcopy(self)

    def get_obs(self, mode="stack_max") -> np.ndarray:
        if mode == "stack_max":
            return np.amax(np.stack(self.buffer, axis=1), axis=1) # leading dim is batch dim [B, 1, C, H, W]
        elif mode == "stack_mean":
            return np.mean(np.stack(self.buffer, axis=1), axis=1, keepdims=True) # leading dim is batch dim [B, 1, C, H, W]
        elif mode == "stack":
            # print ([x.shape for x in self.buffer])
            return np.stack(self.buffer, axis=1) # [B, T, C, H, W]
        else:
            raise NotImplementedError

    def get_fov_locs(self, return_mask=False) -> np.ndarray:
        # print ([x.shape for x in self.fov_loc_buffer])
        return np.stack(self.fov_loc_buffer, axis=1) #[B, T, *fov_locs_size], maybe [B, T, 9]
