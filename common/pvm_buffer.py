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
            if self.fov_loc_size is not None: # (1, 2) or (1, 2, 3)
                self.fov_loc_buffer.append(np.zeros(self.fov_loc_size, dtype=np.float32))

    
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
        elif mode == "stack_channel":
            return np.concatenate(self.buffer, axis=1) # [B, T*C, H, W]
        else:
            raise NotImplementedError

    def get_fov_locs(self, return_mask=False, relative_transform=False) -> np.ndarray:
        # print ([x.shape for x in self.fov_loc_buffer])
        transforms = []
        if relative_transform:
            for t in range(len(self.fov_loc_buffer)):
                transforms.append(np.zeros((self.fov_loc_buffer[t].shape[0], 2, 3), dtype=np.float32)) # [B, 2, 3] B=1 usually
                for b in range(len(self.fov_loc_buffer[t])):
                    extrinsic = self.fov_loc_buffer[t][b]
                    target_extrinsic = self.fov_loc_buffer[t][-1]
                    # print (extrinsic, extrinsic.shape)
                    if np.linalg.det(extrinsic):
                        extrinsic_inv = np.linalg.inv(extrinsic)
                        transform = np.dot(target_extrinsic, extrinsic_inv)
                        # 4x4 transformation matrix
                        R = transform[:3, :3] # Extract the rotation
                        tr = transform[:3, 3] # Extract the translation
                        H = R + np.outer(tr, np.array([0, 0, 1], dtype=np.float32))
                        # Assuming H is the 3x3 homography matrix
                        A = H / H[2, 2]
                        affine = A[:2, :]
                        transforms[t][b] = affine
                    else:
                        H = np.identity(3, dtype=np.float32)
                        A = H / H[2, 2]
                        affine = A[:2, :]
                        transforms[t][b] = affine
            # print (transforms, [x.shape for x in transforms])
            return np.stack(transforms, axis=1) # [B, T, 2, 3]
        else:
            return np.stack(self.fov_loc_buffer, axis=1)
        #[B, T, *fov_locs_size], maybe [B, T, 2] for 2d or [B, T, 4, 4] for 3d
