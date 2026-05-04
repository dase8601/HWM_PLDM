#!/usr/bin/env python3
"""
train_maniskill_hwm.py  —  HWM-style latent world model + SIGReg on FetchPickAndPlace-v4

═══════════════════════════════════════════════════════════════════════════════
INSTALL (RunPod, fresh pod — RTX 4090):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install gymnasium gymnasium-robotics opencv-python-headless numpy tqdm matplotlib

USAGE:
    # Stage 1: offline data collection  (~5–10 min)
    MUJOCO_GL=egl python scripts/train_maniskill_hwm.py --collect

    # Stage 2: train world model        (~30–60 min on 4090)
    MUJOCO_GL=egl python scripts/train_maniskill_hwm.py --train

    # Stage 3: CEM planner evaluation   (~10 min)
    MUJOCO_GL=egl python scripts/train_maniskill_hwm.py --eval

    # All in one:
    MUJOCO_GL=egl python scripts/train_maniskill_hwm.py --collect --train --eval

    # Ablation — no SIGReg:
    MUJOCO_GL=egl python scripts/train_maniskill_hwm.py --collect --train --eval --no-sigreg

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE:

  FetchEncoder   : (B, 3, 64, 64)
                     → Conv2d(32, k=8, s=4) ReLU  → (32, 15, 15)
                     → Conv2d(64, k=4, s=2) ReLU  → (64,  6,  6)
                     → Conv2d(64, k=3, s=1) ReLU  → (64,  4,  4)
                     → Flatten                     → (1024,)
                     → Linear(1024 → 256)
                     → LayerNorm(256)              → (B, 256)

  FetchPredictor : (B, 256+4=260)
                     → Linear(260 → 512) ReLU
                     → Linear(512 → 512) ReLU
                     → Linear(512 → 256)
                     + residual + LayerNorm        → (B, 256)

  WorldModel     : encoder + predictor

LOSSES:
  pred_loss    = MSE(z_pred_t, sg(z_enc_{t+1}))   ×  1.0   [stop-gradient targets]
  sigreg_loss  = (E[proj_mean²] + E[(proj_std-1)²]) × 0.1  [M=1024 random projections]
  vicreg_var   = mean(max(0, γ - std(z)))          × 25.0  [variance floor γ=1]
  vicreg_cov   = sum(off-diag(cov(z))²) / D        ×  1.0  [covariance penalty]

CEM PLANNER (evaluation only):
  H=8 steps, K=200 candidates, top-20 elites, 10 iterations
  cost = cosine_distance(z_predicted_H, z_goal)
  MPC: replan every step

═══════════════════════════════════════════════════════════════════════════════
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import argparse
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

# Set MUJOCO_GL before mujoco is imported — mujoco 3.8+ loads its EGL renderer
# at import time via PyOpenGL, so the env var must be in place beforehand.
# If libEGL.so.1 is missing run:  apt-get install -y libegl1
# Fallback (no system EGL needed): pip install "mujoco==3.1.6" --force-reinstall
os.environ.setdefault("MUJOCO_GL", "egl")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── gymnasium-robotics (registers FetchPickAndPlace-v4) ───────────────────────
try:
    import gymnasium as gym
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
except (ImportError, AttributeError) as e:
    print(f"\n[ERROR] mujoco/gymnasium import failed: {e}")
    if "eglQueryString" in str(e) or "NoneType" in str(e):
        print("\n  Root cause: mujoco 3.8+ requires libEGL.so.1 at import time.")
        print("  Fix 1 (fastest):  apt-get install -y libegl1")
        print("  Fix 2 (fallback): pip install 'mujoco==3.1.6' --force-reinstall")
    else:
        print("  Run:  pip install gymnasium gymnasium-robotics")
    sys.exit(1)

# ── opencv (resize frames) ────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[ERROR] Missing opencv-python-headless")
    print("  Run:  pip install opencv-python-headless")
    sys.exit(1)

# ── optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE     = 64     # pixels, square input to encoder
LATENT_DIM   = 256    # world-model latent dimension D
ACTION_DIM   = 4      # FetchPickAndPlace: [dx, dy, dz, dg] gripper
HIDDEN_DIM   = 512    # predictor hidden units

# Data collection
N_COLLECT_EPISODES = 1000
EP_MAX_STEPS       = 50    # FetchPickAndPlace max steps per episode
SEQ_LEN            = 16   # training context length T

# Training
BATCH_SIZE   = 128
N_EPOCHS     = 150
LR           = 3e-4
WARMUP_STEPS = 500
GRAD_CLIP    = 1.0

# Loss weights
LAMBDA_PRED   = 1.0
LAMBDA_SIGREG = 0.1
N_PROJ        = 1024    # SIGReg random projection count M
LAMBDA_VAR    = 25.0    # VICReg variance coefficient
LAMBDA_COV    = 1.0     # VICReg covariance coefficient
GAMMA_VAR     = 1.0     # target std for variance floor

# CEM planner
CEM_H         = 8
CEM_K         = 200
CEM_ELITES    = 20
CEM_ITERS     = 10
CEM_SIGMA0    = 0.5
CEM_MIN_SIGMA = 0.05

N_EVAL_EPISODES = 50

# Paths
DATA_DIR     = Path("data/fetch_pp")
CKPT_DIR     = Path("checkpoints")
RESULTS_DIR  = Path("results/fetch_pp")


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def make_env(seed: int = 0) -> gym.Env:
    """Create FetchPickAndPlace-v4 with rgb_array renderer (MuJoCo native EGL)."""
    env = gym.make(
        "FetchPickAndPlace-v4",
        render_mode="rgb_array",
        reward_type="dense",
        max_episode_steps=EP_MAX_STEPS,
    )
    env.reset(seed=seed)
    return env


def get_frame(env: gym.Env) -> np.ndarray:
    """Render env → uint8 (IMG_SIZE, IMG_SIZE, 3)."""
    frame = env.render()   # (H, W, 3) uint8 from MuJoCo native renderer
    if frame.shape[0] != IMG_SIZE or frame.shape[1] != IMG_SIZE:
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, H, W) float32 in [0, 1]."""
    return torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPTED P-CONTROLLER  (goal frame collection)
# ══════════════════════════════════════════════════════════════════════════════
#
# FetchPickAndPlace obs_dict['observation'] layout (25-dim):
#   [0:3]   grip_pos        — end-effector xyz
#   [3:6]   object_pos      — object xyz (same as achieved_goal)
#   [6:9]   object_rel_pos  — object_pos - grip_pos
#   [9:13]  object_rot      — quaternion
#   [13:16] object_velp     — linear velocity
#   [16:19] object_velr     — angular velocity
#   [19:22] grip_velp       — end-effector velocity
#   [22:24] gripper_state   — finger widths
#   [24]    gripper_vel     — combined finger velocity (v4 adds this dim)
#
# Action: [dx, dy, dz, dg]  ∈ [-1, 1]
#   dg > 0 = open gripper,  dg < 0 = close gripper

def scripted_pick_and_place(
    env: gym.Env,
    obs_dict: dict,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    3-phase P-controller:
      Phase 1 (15 steps): hover above object, open gripper
      Phase 2 (10 steps): lower onto object, close gripper
      Phase 3 (25 steps): carry to desired_goal, keep closed

    Returns (success, final_frame_uint8).
    """
    HOVER_OFFSET = 0.05   # m above object for phase-1 hover
    KP           = 5.0    # proportional gain

    success = False
    goal_frame = None

    for step_idx in range(EP_MAX_STEPS):
        obs_vec      = obs_dict["observation"]
        grip_pos     = obs_vec[:3]
        object_pos   = obs_dict["achieved_goal"]
        desired_goal = obs_dict["desired_goal"]

        if step_idx < 15:
            # Phase 1: move above object, open
            target     = object_pos + np.array([0.0, 0.0, HOVER_OFFSET])
            action_xyz = np.clip(KP * (target - grip_pos), -1.0, 1.0)
            action     = np.array([*action_xyz, 1.0], dtype=np.float32)

        elif step_idx < 25:
            # Phase 2: lower and grasp, close
            target     = object_pos + np.array([0.0, 0.0, 0.005])
            action_xyz = np.clip(KP * (target - grip_pos), -1.0, 1.0)
            action     = np.array([*action_xyz, -1.0], dtype=np.float32)

        else:
            # Phase 3: carry to goal, keep closed
            target     = desired_goal
            action_xyz = np.clip(KP * (target - grip_pos), -1.0, 1.0)
            action     = np.array([*action_xyz, -1.0], dtype=np.float32)

        obs_dict, reward, terminated, truncated, info = env.step(action)

        if info.get("is_success", False):
            success    = True
            goal_frame = get_frame(env)
            break

        if terminated or truncated:
            break

    if goal_frame is None:
        goal_frame = get_frame(env)   # best-effort final frame

    return success, goal_frame


# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

class Episode:
    """Single episode: frames[0..T], actions[0..T-1], rewards[0..T-1]."""
    def __init__(self):
        self.frames  : List[np.ndarray] = []   # (H, W, 3) uint8
        self.actions : List[np.ndarray] = []   # (4,) float32
        self.rewards : List[float]      = []
        self.success : bool             = False


def collect_episodes(n_episodes: int, seed_offset: int = 0) -> List[Episode]:
    """
    Collect n_episodes with mixed policy:
      - First 5% + random 5%: scripted P-controller (ensures goal coverage)
      - Rest: uniform random actions

    Each episode has at least SEQ_LEN frames to be kept.
    """
    env = make_env(seed=seed_offset)
    episodes  = []
    successes = 0

    pbar = tqdm(total=n_episodes, desc="[COLLECT] episodes")

    for ep_idx in range(n_episodes * 2):   # extra budget for short episodes
        if len(episodes) >= n_episodes:
            break

        seed       = seed_offset + ep_idx
        obs_dict, _ = env.reset(seed=seed)

        ep = Episode()
        ep.frames.append(get_frame(env))

        use_scripted = (ep_idx < n_episodes // 20) or (random.random() < 0.05)

        if use_scripted:
            # ── Scripted P-controller ─────────────────────────────────────
            HOVER_OFFSET = 0.05
            KP           = 5.0

            for step_idx in range(EP_MAX_STEPS):
                obs_vec      = obs_dict["observation"]
                grip_pos     = obs_vec[:3]
                object_pos   = obs_dict["achieved_goal"]
                desired_goal = obs_dict["desired_goal"]

                if step_idx < 15:
                    target     = object_pos + np.array([0.0, 0.0, HOVER_OFFSET])
                    action_xyz = np.clip(KP * (target - grip_pos), -1.0, 1.0)
                    action     = np.array([*action_xyz,  1.0], dtype=np.float32)
                elif step_idx < 25:
                    target     = object_pos + np.array([0.0, 0.0, 0.005])
                    action_xyz = np.clip(KP * (target - grip_pos), -1.0, 1.0)
                    action     = np.array([*action_xyz, -1.0], dtype=np.float32)
                else:
                    target     = desired_goal
                    action_xyz = np.clip(KP * (target - grip_pos), -1.0, 1.0)
                    action     = np.array([*action_xyz, -1.0], dtype=np.float32)

                obs_dict, reward, terminated, truncated, info = env.step(action)
                ep.frames.append(get_frame(env))
                ep.actions.append(action)
                ep.rewards.append(float(reward))

                if info.get("is_success", False):
                    ep.success = True
                    successes += 1
                    break
                if terminated or truncated:
                    break
        else:
            # ── Random policy ─────────────────────────────────────────────
            for _ in range(EP_MAX_STEPS):
                action = env.action_space.sample().astype(np.float32)
                obs_dict, reward, terminated, truncated, info = env.step(action)
                ep.frames.append(get_frame(env))
                ep.actions.append(action)
                ep.rewards.append(float(reward))
                if info.get("is_success", False):
                    ep.success = True
                    successes += 1
                    break
                if terminated or truncated:
                    break

        # Require at least SEQ_LEN frames
        if len(ep.frames) >= SEQ_LEN:
            episodes.append(ep)
            pbar.update(1)
            pbar.set_postfix({
                "success": successes,
                "rate":    f"{100*successes/max(1,len(episodes)):.1f}%",
            })

    pbar.close()
    env.close()

    print(
        f"\n[COLLECT] Done. {len(episodes)} usable episodes | "
        f"{successes} P-controller successes "
        f"({100*successes/max(1,len(episodes)):.1f}%)"
    )
    return episodes[:n_episodes]


def save_episodes(episodes: List[Episode], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "frames" : [ep.frames  for ep in episodes],
        "actions": [ep.actions for ep in episodes],
        "rewards": [ep.rewards for ep in episodes],
        "success": [ep.success for ep in episodes],
    }
    out = path / "episodes.pt"
    torch.save(data, out)
    print(f"[COLLECT] Saved {len(episodes)} episodes → {out}")


def load_episodes(path: Path) -> List[Episode]:
    src  = path / "episodes.pt"
    data = torch.load(src, weights_only=False)
    eps  = []
    for i in range(len(data["frames"])):
        ep         = Episode()
        ep.frames  = data["frames"][i]
        ep.actions = data["actions"][i]
        ep.rewards = data["rewards"][i]
        ep.success = data["success"][i]
        eps.append(ep)
    print(f"[DATA] Loaded {len(eps)} episodes from {src}")
    return eps


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset over collected episodes.

    __getitem__ returns:
        obs_seq    : (SEQ_LEN, 3, IMG_SIZE, IMG_SIZE)  float32 [0,1]
        action_seq : (SEQ_LEN-1, ACTION_DIM)           float32 [-1,1]
    """

    def __init__(self, episodes: List[Episode], seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.windows : List[Tuple[Episode, int]] = []   # (episode, start_t)

        stride = max(1, seq_len // 4)   # 25% overlap between windows
        for ep in episodes:
            T = len(ep.frames)
            if T < seq_len:
                continue
            for start in range(0, T - seq_len + 1, stride):
                # Ensure actions exist for every obs transition in window
                if start + seq_len - 2 < len(ep.actions):
                    self.windows.append((ep, start))

        print(
            f"[DATASET] {len(self.windows)} windows "
            f"from {len(episodes)} episodes "
            f"(seq_len={seq_len}, stride={stride})"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        ep, start = self.windows[idx]
        end       = start + self.seq_len

        obs_seq    = torch.stack([frame_to_tensor(ep.frames[t])
                                  for t in range(start, end)])           # (T, 3, H, W)
        action_seq = torch.stack([torch.tensor(ep.actions[t], dtype=torch.float32)
                                  for t in range(start, end - 1)])       # (T-1, A)
        return obs_seq, action_seq


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class FetchEncoder(nn.Module):
    """
    DQN-style CNN: (B, 3, 64, 64) → (B, LATENT_DIM)

    Spatial sizes (no padding):
        Input  : (3,  64, 64)
        Conv32 : (32, 15, 15)   [k=8 s=4 → floor((64-8)/4)+1 = 15]
        Conv64 : (64,  6,  6)   [k=4 s=2 → floor((15-4)/2)+1 =  6]
        Conv64 : (64,  4,  4)   [k=3 s=1 → floor(( 6-3)/1)+1 =  4]
        Flatten: (1024,)
        Linear : (LATENT_DIM,)
        LN     : (LATENT_DIM,)
    """
    _CONV_FLAT = 64 * 4 * 4   # = 1024

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.fc   = nn.Linear(self._CONV_FLAT, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, D)"""
        h = self.conv(x)            # (B, 64, 4, 4)
        h = h.flatten(start_dim=1)  # (B, 1024)
        h = self.fc(h)              # (B, D)
        return self.norm(h)

    def forward_seq(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq: (T, B, 3, H, W) → (T, B, D)"""
        T, B, C, H, W = obs_seq.shape
        z = self.forward(obs_seq.reshape(T * B, C, H, W))   # (T*B, D)
        return z.reshape(T, B, self.latent_dim)


class FetchPredictor(nn.Module):
    """
    Latent transition MLP with residual connection.
    (B, D) × (B, A) → (B, D)

    z_cat = cat(z, a)                        # (B, D+A)
    delta = Linear(D+A→H) ReLU Linear(H→H) ReLU Linear(H→D)
    out   = LayerNorm(z + delta)             # residual
    """

    def __init__(
        self,
        latent_dim:  int = LATENT_DIM,
        action_dim:  int = ACTION_DIM,
        hidden_dim:  int = HIDDEN_DIM,
    ):
        super().__init__()
        in_dim = latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, D)  a: (B, A) → (B, D)"""
        delta = self.net(torch.cat([z, a], dim=-1))
        return self.norm(z + delta)


class WorldModel(nn.Module):
    """Encoder + recurrent predictor."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.encoder   = FetchEncoder(latent_dim=latent_dim)
        self.predictor = FetchPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def forward(
        self,
        obs_seq:    torch.Tensor,   # (T, B, 3, H, W)
        action_seq: torch.Tensor,   # (T-1, B, A)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_enc  : (T,   B, D)  — encoder outputs for all timesteps
            z_pred : (T-1, B, D)  — predictor outputs for timesteps 1…T-1
        """
        z_enc = self.encoder.forward_seq(obs_seq)   # (T, B, D)

        z_preds = []
        z = z_enc[0]
        for t in range(obs_seq.shape[0] - 1):
            z = self.predictor(z, action_seq[t])
            z_preds.append(z)

        z_pred = torch.stack(z_preds, dim=0)        # (T-1, B, D)
        return z_enc, z_pred


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def prediction_loss(
    z_pred:   torch.Tensor,   # (T-1, B, D)
    z_target: torch.Tensor,   # (T-1, B, D)
) -> torch.Tensor:
    """MSE with stop-gradient on encoder targets."""
    return F.mse_loss(z_pred, z_target.detach())


def sigreg_loss(
    z: torch.Tensor,          # (N, D)
    n_proj: int = N_PROJ,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SIGReg: enforce z ~ N(0, I) via M random unit-norm projections.

    Projections w ~ S^{D-1} (unit sphere).
    mean_loss = E_w[ (E_n[w·z_n])² ]    → push projected mean  → 0
    std_loss  = E_w[ (std_n[w·z_n]-1)² ] → push projected std  → 1

    Ref: spectral isotropic Gaussian regularization.
    """
    N, D = z.shape
    W    = torch.randn(D, n_proj, device=z.device, dtype=z.dtype)
    W    = F.normalize(W, dim=0)                         # unit-norm columns
    proj = z @ W                                         # (N, n_proj)
    mean_loss = proj.mean(dim=0).pow(2).mean()
    std_loss  = (proj.std(dim=0) - 1.0).pow(2).mean()
    return mean_loss, std_loss


def vicreg_loss(
    z: torch.Tensor,          # (N, D)
    lambda_var: float = LAMBDA_VAR,
    lambda_cov: float = LAMBDA_COV,
    gamma: float      = GAMMA_VAR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    VICReg variance + covariance regularization.

    var_loss : penalises any dimension whose std falls below gamma
    cov_loss : penalises off-diagonal entries of the sample covariance
    """
    N, D = z.shape
    z_c  = z - z.mean(dim=0, keepdim=True)          # (N, D) centred

    std      = z.std(dim=0)                          # (D,)
    var_loss = F.relu(gamma - std).mean()

    cov      = (z_c.T @ z_c) / (N - 1)              # (D, D)
    off_diag = cov.masked_fill(torch.eye(D, dtype=torch.bool, device=z.device), 0.0)
    cov_loss = off_diag.pow(2).sum() / D

    return lambda_var * var_loss, lambda_cov * cov_loss


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model:       WorldModel,
    dataloader:  DataLoader,
    device:      torch.device,
    n_epochs:    int   = N_EPOCHS,
    lr:          float = LR,
    use_sigreg:  bool  = True,
    run_name:    str   = "hwm_fetch",
    ckpt_dir:    Path  = CKPT_DIR,
    results_dir: Path  = RESULTS_DIR,
) -> dict:

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = n_epochs * len(dataloader)

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # EWA trackers (α=0.05 → ~20-step memory)
    ewa   = dict(loss=0.0, pred=0.0, sigreg=0.0, var=0.0, cov=0.0)
    alpha = 0.05
    history = {k: [] for k in ewa}
    history["epoch_loss"] = []

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Training : {run_name}")
    print(f"  Epochs   : {n_epochs}  |  Steps/epoch : {len(dataloader)}")
    print(f"  Batch    : {BATCH_SIZE}  |  LR : {lr}  |  Device : {device}")
    print(f"  SIGReg   : {'ON  λ=%.2f  M=%d' % (LAMBDA_SIGREG, N_PROJ) if use_sigreg else 'OFF (ablation)'}")
    print(f"  VICReg   : var=ON λ={LAMBDA_VAR}  cov=ON λ={LAMBDA_COV}")
    print(f"{'═'*70}\n")

    best_loss  = float("inf")
    gstep      = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for bidx, (obs_seq, action_seq) in enumerate(dataloader):
            # DataLoader gives (B, T, …) — transpose to (T, B, …)
            obs_seq    = obs_seq.to(device).permute(1, 0, 2, 3, 4)  # (T, B, 3, H, W)
            action_seq = action_seq.to(device).permute(1, 0, 2)     # (T-1, B, A)

            z_enc, z_pred = model(obs_seq, action_seq)
            # z_enc:  (T,   B, D)
            # z_pred: (T-1, B, D)

            T, B, D = z_enc.shape

            # ── 1. Prediction loss (stop-gradient on encoder targets) ────────
            pred_l = prediction_loss(z_pred, z_enc[1:])
            loss   = LAMBDA_PRED * pred_l

            # ── 2. SIGReg ────────────────────────────────────────────────────
            z_flat = z_enc.reshape(T * B, D)
            if use_sigreg:
                sig_mean, sig_std = sigreg_loss(z_flat)
                sigreg_l = LAMBDA_SIGREG * (sig_mean + sig_std)
                loss     = loss + sigreg_l
            else:
                sigreg_l = torch.zeros(1, device=device)

            # ── 3. VICReg ────────────────────────────────────────────────────
            vic_var, vic_cov = vicreg_loss(z_flat)
            loss = loss + vic_var + vic_cov

            # ── Backward ─────────────────────────────────────────────────────
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            gstep += 1

            # ── EWA bookkeeping ───────────────────────────────────────────────
            vals = dict(
                loss   = loss.item(),
                pred   = pred_l.item(),
                sigreg = sigreg_l.item() if use_sigreg else 0.0,
                var    = vic_var.item(),
                cov    = vic_cov.item(),
            )
            for k, v in vals.items():
                ewa[k] = alpha * v + (1.0 - alpha) * ewa[k]
                history[k].append(v)
            epoch_losses.append(loss.item())

            # ── Per-step log every 25 batches ─────────────────────────────────
            if (bidx + 1) % 25 == 0:
                lr_now = sched.get_last_lr()[0]
                print(
                    f"  [ep {epoch+1:3d}/{n_epochs}"
                    f" | step {bidx+1:4d}/{len(dataloader)}]"
                    f"  loss={ewa['loss']:.4f}"
                    f"  pred={ewa['pred']:.4f}"
                    f"  sig={ewa['sigreg']:.4f}"
                    f"  var={ewa['var']:.4f}"
                    f"  cov={ewa['cov']:.4f}"
                    f"  lr={lr_now:.2e}"
                )

        # ── Epoch summary ──────────────────────────────────────────────────────
        mean_loss = float(np.mean(epoch_losses))
        history["epoch_loss"].append(mean_loss)
        elapsed = time.time() - t0

        print(
            f"\n  ── Epoch {epoch+1:3d}/{n_epochs} ──"
            f"  mean_loss={mean_loss:.4f}"
            f"  time={elapsed:.1f}s"
            f"\n  EWA: pred={ewa['pred']:.4f}"
            f"  sig={ewa['sigreg']:.4f}"
            f"  var={ewa['var']:.4f}"
            f"  cov={ewa['cov']:.4f}\n"
        )

        # ── Checkpoint (best) ──────────────────────────────────────────────────
        if mean_loss < best_loss:
            best_loss = mean_loss
            p = ckpt_dir / f"{run_name}_best.pt"
            torch.save(dict(
                epoch=epoch, model=model.state_dict(),
                optimizer=opt.state_dict(), loss=best_loss,
                use_sigreg=use_sigreg,
            ), p)
            print(f"  ★ Best checkpoint → {p}  (loss={best_loss:.4f})\n")

        if (epoch + 1) % 50 == 0:
            p = ckpt_dir / f"{run_name}_ep{epoch+1}.pt"
            torch.save(dict(epoch=epoch, model=model.state_dict()), p)
            print(f"  Checkpoint → {p}\n")

    # ── Final checkpoint ───────────────────────────────────────────────────────
    p = ckpt_dir / f"{run_name}_final.pt"
    torch.save(dict(epoch=n_epochs - 1, model=model.state_dict()), p)
    print(f"\n[TRAIN] Final → {p}")

    # ── Loss curves ────────────────────────────────────────────────────────────
    if HAS_MPL:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        ax = axes.flatten()
        for i, k in enumerate(["loss", "pred", "sigreg", "var", "cov"]):
            ax[i].plot(history[k])
            ax[i].set_title(k)
            ax[i].set_xlabel("batch")
        ax[5].plot(history["epoch_loss"])
        ax[5].set_title("epoch loss")
        ax[5].set_xlabel("epoch")
        plt.suptitle(run_name)
        plt.tight_layout()
        fig_p = results_dir / f"{run_name}_loss_curves.png"
        plt.savefig(fig_p, dpi=120)
        plt.close()
        print(f"[TRAIN] Loss curves → {fig_p}")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# CEM PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class CEMPlanner:
    """
    Cross-Entropy Method in latent space.

    Each call to plan():
      1. Initialise Gaussian N(mu, sigma² I) over H×A actions
      2. Sample K action sequences, clamp to [-1, 1]
      3. Roll out world model predictor for H steps
      4. Cost = cosine_distance(z_H, z_goal)   ∈ [0, 2]
      5. Select top-e elite sequences, update mu and sigma
      6. Repeat for n_iters
      7. Return mu[0] — first action of the best plan
    """

    def __init__(
        self,
        model:     WorldModel,
        H:         int   = CEM_H,
        K:         int   = CEM_K,
        elites:    int   = CEM_ELITES,
        iters:     int   = CEM_ITERS,
        sigma0:    float = CEM_SIGMA0,
        min_sigma: float = CEM_MIN_SIGMA,
        device:    torch.device = torch.device("cpu"),
    ):
        self.model     = model
        self.H         = H
        self.K         = K
        self.elites    = elites
        self.iters     = iters
        self.sigma0    = sigma0
        self.min_sigma = min_sigma
        self.device    = device
        self.A         = model.action_dim

    @torch.no_grad()
    def plan(self, z_curr: torch.Tensor, z_goal: torch.Tensor) -> np.ndarray:
        """
        z_curr : (D,)  current latent state
        z_goal : (D,)  goal latent state
        Returns: (A,)  first action from the best plan
        """
        z0 = z_curr.to(self.device).unsqueeze(0).expand(self.K, -1)  # (K, D)
        zg = z_goal.to(self.device).unsqueeze(0)                       # (1, D)

        mu    = torch.zeros(self.H, self.A, device=self.device)
        sigma = torch.full((self.H, self.A), self.sigma0, device=self.device)

        for _ in range(self.iters):
            # Sample K×H×A actions from current distribution
            noise   = torch.randn(self.K, self.H, self.A, device=self.device)
            actions = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise).clamp(-1.0, 1.0)
            # actions : (K, H, A)

            # Roll out H steps
            z = z0.clone()               # (K, D)
            for h in range(self.H):
                z = self.model.predictor(z, actions[:, h, :])   # (K, D)
            # z : (K, D)  final predicted state

            # Cost: cosine distance to goal
            cos_sim = F.cosine_similarity(z, zg.expand_as(z), dim=-1)  # (K,)
            cost    = 1.0 - cos_sim                                      # (K,)

            # Elite update
            elite_idx     = cost.argsort()[:self.elites]    # (elites,)
            elite_actions = actions[elite_idx]              # (elites, H, A)
            mu    = elite_actions.mean(dim=0)               # (H, A)
            sigma = elite_actions.std(dim=0).clamp(min=self.min_sigma)

        return mu[0].cpu().numpy()   # (A,) first action in best plan


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def collect_goal_frames(
    n_goals: int,
    device: torch.device,
    seed_offset: int = 9000,
) -> List[torch.Tensor]:
    """
    Run P-controller episodes; collect the final frame from each
    attempt as a goal image (success or not — best-effort).
    Returns list of (3, H, W) float tensors.
    """
    env    = make_env(seed=seed_offset)
    goals  = []
    budget = n_goals * 3
    pbar   = tqdm(total=n_goals, desc="[EVAL] Collecting goal frames")

    for attempt in range(budget):
        if len(goals) >= n_goals:
            break
        obs_dict, _ = env.reset(seed=seed_offset + attempt)
        success, goal_frame = scripted_pick_and_place(env, obs_dict)
        if goal_frame is not None:
            goals.append(frame_to_tensor(goal_frame))
            pbar.update(1)

    pbar.close()
    env.close()
    print(f"[EVAL] Collected {len(goals)} goal frames ({budget} attempts)")
    return goals


def evaluate(
    model:       WorldModel,
    device:      torch.device,
    n_episodes:  int   = N_EVAL_EPISODES,
    run_name:    str   = "hwm_fetch",
    results_dir: Path  = RESULTS_DIR,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    planner = CEMPlanner(model=model, device=device)

    print(f"\n{'═'*70}")
    print(f"  Evaluating : {run_name}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  CEM        : H={CEM_H}  K={CEM_K}  elites={CEM_ELITES}  iters={CEM_ITERS}")
    print(f"{'═'*70}\n")

    # Pre-collect + encode goal frames
    goal_tensors = collect_goal_frames(
        n_goals=max(20, n_episodes // 5),
        device=device,
    )
    if not goal_tensors:
        print("[EVAL ERROR] No goal frames collected — P-controller failure.")
        return {"success_rate": 0.0, "mean_reward": 0.0}

    with torch.no_grad():
        z_goals = model.encoder(
            torch.stack(goal_tensors).to(device)     # (G, 3, H, W)
        )                                             # (G, D)

    env        = make_env(seed=42)
    successes  = 0
    ep_rewards = []

    pbar = tqdm(range(n_episodes), desc="[EVAL] episodes")
    for ep_idx in pbar:
        obs_dict, _ = env.reset(seed=42 + ep_idx)

        z_goal   = z_goals[ep_idx % len(z_goals)]   # (D,)
        ep_rew   = 0.0
        ep_ok    = False

        best_action = np.zeros(ACTION_DIM, dtype=np.float32)

        for step in range(EP_MAX_STEPS):
            # Encode current frame
            frame_t = frame_to_tensor(get_frame(env)).unsqueeze(0).to(device)  # (1,3,H,W)
            with torch.no_grad():
                z_curr = model.encoder(frame_t).squeeze(0)    # (D,)

            # CEM plan (every step = full MPC)
            best_action = planner.plan(z_curr, z_goal)
            action      = best_action.clip(-1.0, 1.0)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward

            if info.get("is_success", False):
                ep_ok = True
                break
            if terminated or truncated:
                break

        successes  += int(ep_ok)
        ep_rewards.append(ep_rew)

        pbar.set_postfix({
            "ok":   f"{successes}/{ep_idx+1}",
            "rate": f"{100*successes/max(1,ep_idx+1):.1f}%",
        })

        print(
            f"  ep {ep_idx+1:3d}"
            f"  success={'YES' if ep_ok else ' no'}"
            f"  reward={ep_rew:7.2f}"
            f"  cumrate={100*successes/max(1,ep_idx+1):.1f}%"
        )

    env.close()

    sr   = successes / max(1, n_episodes)
    mr   = float(np.mean(ep_rewards))

    print(f"\n{'═'*70}")
    print(f"  RESULTS: {run_name}")
    print(f"  Success  : {100*sr:.1f}%  ({successes}/{n_episodes})")
    print(f"  Reward   : {mr:.2f} ± {float(np.std(ep_rewards)):.2f}")
    print(f"{'═'*70}\n")

    results = dict(
        run_name     = run_name,
        success_rate = sr,
        successes    = successes,
        n_episodes   = n_episodes,
        mean_reward  = mr,
        std_reward   = float(np.std(ep_rewards)),
        rewards      = ep_rewards,
    )

    p = results_dir / f"{run_name}_eval.json"
    with open(p, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "rewards"}, f, indent=2)
    print(f"[EVAL] Results → {p}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_model() -> WorldModel:
    return WorldModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HWM + SIGReg world model on FetchPickAndPlace-v4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--collect",    action="store_true", help="Stage 1: collect offline data")
    parser.add_argument("--train",      action="store_true", help="Stage 2: train world model")
    parser.add_argument("--eval",       action="store_true", help="Stage 3: CEM evaluation")
    parser.add_argument("--no-sigreg",  action="store_true", help="Ablation: disable SIGReg")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name",   default="hwm_fetch")
    parser.add_argument("--n-collect",  type=int,   default=N_COLLECT_EPISODES)
    parser.add_argument("--n-epochs",   type=int,   default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--n-eval",     type=int,   default=N_EVAL_EPISODES)
    parser.add_argument("--ckpt",       default=None, help="Checkpoint path for --eval")
    args = parser.parse_args()

    if not (args.collect or args.train or args.eval):
        parser.print_help()
        sys.exit(0)

    device     = torch.device(args.device)
    use_sigreg = not args.no_sigreg
    run_name   = args.run_name + ("" if use_sigreg else "_no_sigreg")

    print(f"\n{'#'*70}")
    print(f"  HWM + {'SIGReg' if use_sigreg else 'baseline (no SIGReg)'}")
    print(f"  run_name  : {run_name}")
    print(f"  device    : {device}")
    mgl = os.environ.get("MUJOCO_GL", "NOT SET")
    print(f"  MUJOCO_GL : {mgl}" + (" ← set to 'egl' on headless pods" if mgl == "NOT SET" else ""))
    print(f"{'#'*70}\n")

    # ── STAGE 1: COLLECT ──────────────────────────────────────────────────────
    if args.collect:
        print("═" * 40)
        print("  STAGE 1: DATA COLLECTION")
        print("═" * 40)
        eps = collect_episodes(n_episodes=args.n_collect)
        save_episodes(eps, DATA_DIR)

    # ── STAGE 2: TRAIN ────────────────────────────────────────────────────────
    if args.train:
        print("═" * 40)
        print("  STAGE 2: TRAINING")
        print("═" * 40)

        ep_file = DATA_DIR / "episodes.pt"
        if not ep_file.exists():
            print(f"[ERROR] No data at {ep_file}.  Run --collect first.")
            sys.exit(1)

        eps     = load_episodes(DATA_DIR)
        dataset = TrajectoryDataset(eps, seq_len=SEQ_LEN)
        loader  = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = True,
            num_workers = 4,
            pin_memory  = (args.device == "cuda"),
            drop_last   = True,
        )

        model = build_model()
        print(f"\n[MODEL] WorldModel  : {count_params(model):>9,} params")
        print(f"        FetchEncoder : {count_params(model.encoder):>9,} params")
        print(f"        FetchPredictor:{count_params(model.predictor):>9,} params")

        print(f"""
[ARCH] FetchEncoder
       ({3},{IMG_SIZE},{IMG_SIZE})
         → Conv2d(32, k=8, s=4) ReLU  → (32, 15, 15)
         → Conv2d(64, k=4, s=2) ReLU  → (64,  6,  6)
         → Conv2d(64, k=3, s=1) ReLU  → (64,  4,  4)
         → Flatten(1024) → Linear → LayerNorm → ({LATENT_DIM},)

[ARCH] FetchPredictor
       ({LATENT_DIM}+{ACTION_DIM}={LATENT_DIM+ACTION_DIM},) → Linear({HIDDEN_DIM}) ReLU
         → Linear({HIDDEN_DIM}) ReLU → Linear({LATENT_DIM})
         → z + delta  (residual) → LayerNorm → ({LATENT_DIM},)

[LOSS] pred_loss    = MSE(z_pred, sg(z_enc))           × {LAMBDA_PRED}
       sigreg_loss  = (mean→0 + std→1 on {N_PROJ} projs) × {LAMBDA_SIGREG}  {'[ACTIVE]' if use_sigreg else '[DISABLED]'}
       vicreg_var   = max(0, γ-std(z))                 × {LAMBDA_VAR}
       vicreg_cov   = off-diag(cov(z))²/D              × {LAMBDA_COV}
""")

        train(
            model       = model,
            dataloader  = loader,
            device      = device,
            n_epochs    = args.n_epochs,
            lr          = args.lr,
            use_sigreg  = use_sigreg,
            run_name    = run_name,
            ckpt_dir    = CKPT_DIR,
            results_dir = RESULTS_DIR,
        )

    # ── STAGE 3: EVAL ─────────────────────────────────────────────────────────
    if args.eval:
        print("═" * 40)
        print("  STAGE 3: EVALUATION")
        print("═" * 40)

        model = build_model()

        ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIR / f"{run_name}_best.pt"
        if not ckpt_path.exists():
            fallback = CKPT_DIR / f"{run_name}_final.pt"
            if fallback.exists():
                ckpt_path = fallback
                print(f"[EVAL] No best.pt — using {fallback}")
            else:
                print(f"[ERROR] No checkpoint at {ckpt_path}.  Run --train first.")
                sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[EVAL] Loaded {ckpt_path}  epoch={ckpt.get('epoch','?')}  loss={ckpt.get('loss','N/A')}")

        results = evaluate(
            model       = model,
            device      = device,
            n_episodes  = args.n_eval,
            run_name    = run_name,
            results_dir = RESULTS_DIR,
        )

        print(f"\n[FINAL] {run_name}")
        print(f"        success_rate = {100*results['success_rate']:.1f}%")
        print(f"        mean_reward  = {results['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
