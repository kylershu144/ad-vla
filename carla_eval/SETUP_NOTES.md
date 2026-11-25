# AutoVLA-in-CARLA setup notes (handoff)

Last updated 2026-11-25. Goal: closed-loop evaluation of AutoVLA in CARLA 1.0, the first step toward
patch generation (see `../cs283_project_plan.md`).

## Where things stand

| Piece | Status |
|---|---|
| AutoVLA inference (`autovla_policy.py`) | Works. Tested on a V100 in fp16: about 4-5 s per call (fast-thinking), 8.3 GB peak VRAM, sensible trajectories |
| Eval harness (`run_eval.py`) | Written, **not yet run against a server** |
| CARLA 1.0 server | **Can't run on a V100.** Moving to an NVIDIA L4 |

### Why the V100 doesn't work
The CARLA 1.0 package (UE 5.8) ships shaders for `VULKAN_SM6` only and checks the GPU against UE's
`VP_UE_Vulkan_SM6` Vulkan profile. On the V100 (Volta) it fails with *"Vulkan device could not be created at
the project's supported feature levels"*. The driver lacks mesh shaders, VRS, and `VK_EXT_mutable_descriptor_type`.
There is no fallback. An L4 (Ada) meets the profile, has 24 GB of VRAM (enough for CARLA and AutoVLA together), and supports bf16.

### Decisions
- **No Bench2Drive.** It is pinned to CARLA 0.9.15, and the paper's Bench2Drive checkpoint isn't public. Use the custom
  harness here. If standardized final numbers are wanted later, the official Leaderboard's `ue58-dev` branch
  targets CARLA 1.0 (Town12/Town13 routes).
- The only public checkpoint (`Zewei-Zhou/AutoVLA`, `AutoVLA_PDMS_89.ckpt`) is trained on nuPlan/NAVSIM, so
  CARLA results are **zero-shot transfer** and won't match the paper's CARLA numbers.

## Rebuilding on a new VM (L4)

1. **Repo:** `git submodule update --init`. The `.gitmodules` path was fixed to `autovla`.
2. **Driver:** check `nvidia-smi`. For rendering also install `libnvidia-gl-<ver>-server`, `libvulkan1`,
   `vulkan-tools`, and the libglvnd runtime (`libegl1 libgl1 libgles2 libglx0`). The NVIDIA Vulkan ICD
   dlopens `libEGL.so.1` and fails with `VK_ERROR_INCOMPATIBLE_DRIVER` without it.
   Verify with `vulkaninfo --summary`, which should list the NVIDIA GPU, not only llvmpipe.
   On the V100 VM, installing the GL package set off a udev load/unload loop of the nvidia modules
   (`/lib/udev/rules.d/71-nvidia.rules`). A reboot cleared it.
3. **Python:** `apt install python3.10-venv`, then
   `python3 -m venv .venv && .venv/bin/pip install uv`, then
   `.venv/bin/uv pip install --python .venv/bin/python torch==2.4.0 torchvision==0.19.0 transformers==4.49.0 tokenizers==0.21.1 accelerate==1.5.2 qwen-vl-utils==0.0.10 safetensors==0.5.3 peft pyyaml "numpy<2" opencv-python-headless==4.9.0.80 pillow av==14.2.0 huggingface-hub==0.29.3 scipy shapely networkx pytorch-lightning==2.2.1`
   and the CARLA wheel `~/carla/Carla-1.0-Linux-Shipping/PythonAPI/carla/dist/carla-1.0-cp310-*.whl`.
4. **Models** (into `models/`, gitignored):
   - `AutoVLA_PDMS_89.ckpt` from `https://huggingface.co/Zewei-Zhou/AutoVLA/resolve/main/AutoVLA_PDMS_89.ckpt` (16 GB).
     The first load writes an fp16 safetensors cache next to it.
   - `Qwen/Qwen2.5-VL-3B-Instruct` into `models/Qwen2.5-VL-3B-Instruct`. This supplies the processor and architecture.
5. **CARLA 1.0:** `https://downloads.carlasim.com/Linux/CARLA_1.0.tar.gz` (14 GB), extracted to `~/carla/`.
   Start the server with
   `CARLA_PACKAGE_ROOT=~/carla/Carla-1.0-Linux-Shipping DETACH=1 bash ~/carla/Carla-1.0-Linux-Shipping/Skills/ue58/run-carla-ue58-server/scripts/run_server.sh package`.
   - Shipping builds write no log. If it dies, run
     `CarlaUnreal/Binaries/Linux/CarlaUnreal-Linux-Shipping CarlaUnreal -RenderOffScreen -stdout -FullStdOutLogOutput` in the foreground.
   - A package only loads its cooked maps (`get_available_maps()`), and the map argument on the command line is ignored, so call `load_world()` after connecting.
   - The first launch compiles shaders and can take longer than the script's 120 s wait.
6. **On an L4:** switch `dtype` in `autovla_policy.py` to `torch.bfloat16` (the model's native dtype; fp16 was only needed on the V100).

## Next steps
1. Harness test without the model: `.venv/bin/python carla_eval/run_eval.py --dry-policy --routes 1 --out results/dry`.
   Expect to fix CARLA 1.0 API differences: blueprint attributes, `agents.navigation` import path (`PythonAPI/carla/agents`
   must be on `PYTHONPATH`), and the traffic-light check.
2. Clean baseline: `.venv/bin/python carla_eval/run_eval.py --routes 10 --save-images --out results/clean`.
3. Then patch generation (Method 1, SPSA), using the logs (`log.json` includes per-step action-token log-probs).

## How the harness feeds AutoVLA (matching its nuPlan training)
- Three cameras: front (FOV 64°), front-left and front-right at ±55° yaw, mounted at 1.7 m on `vehicle.lincoln.mkz`
  (nuPlan's car), 960x540, saved as JPEG.
- 4 frames per camera at 2 Hz, speed and acceleration magnitudes, and a command from the route: `turn left` / `turn right` / `keep forward`.
- Output: 10 poses at 0.5 s steps in the ego frame (x forward, y left). Pure-pursuit and PID control track them, with the policy called every 0.5 s at 20 Hz sim.
- Metrics (Leaderboard definitions): Route Completion x infraction penalties (pedestrian collision 0.5, vehicle 0.6, static 0.65,
  red light 0.7). Off-road distance is subtracted from Route Completion.
