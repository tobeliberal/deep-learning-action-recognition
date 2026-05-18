import numpy as np
from typing import Optional, Union


class OneEuroFilter:
    def __init__(
        self,
        num_keypoints: int = 17,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        freq: float = 30.0
    ):
        self.num_keypoints = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        
        self._prev_values: Optional[np.ndarray] = None
        self._prev_dx: Optional[np.ndarray] = None
        self._x_filtered: Optional[np.ndarray] = None
        self._dx_filtered: Optional[np.ndarray] = None
        self._initialized = False
    
    def _compute_alpha(self, cutoff: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def update(self, current_coords: np.ndarray, confidence: Optional[np.ndarray] = None) -> np.ndarray:
        coords = np.asarray(current_coords, dtype=np.float32)
        
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        
        if confidence is None and coords.shape[1] >= 3:
            confidence = coords[:, 2]
            xy = coords[:, :2]
        else:
            xy = coords[:, :2] if coords.shape[1] >= 2 else coords
            confidence = confidence if confidence is not None else np.ones(len(xy))
        
        if not self._initialized:
            self._prev_values = xy.copy()
            self._x_filtered = xy.copy()
            self._prev_dx = np.zeros_like(xy)
            self._dx_filtered = np.zeros_like(xy)
            self._initialized = True
            return coords.copy()
        
        dx = xy - self._prev_values
        
        alpha_d = self._compute_alpha(self.d_cutoff)
        self._dx_filtered = alpha_d * dx + (1.0 - alpha_d) * self._dx_filtered
        
        speed = np.abs(self._dx_filtered)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._compute_alpha(cutoff)
        
        self._x_filtered = alpha * xy + (1.0 - alpha) * self._x_filtered
        
        low_conf_mask = confidence < 0.1
        self._x_filtered[low_conf_mask] = xy[low_conf_mask]
        
        self._prev_values = xy.copy()
        
        result = self._x_filtered.copy()
        
        if coords.shape[1] >= 3:
            return np.column_stack([result, coords[:, 2:]])
        
        return result
    
    def reset(self):
        self._prev_values = None
        self._prev_dx = None
        self._x_filtered = None
        self._dx_filtered = None
        self._initialized = False
