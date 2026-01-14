# NCS Visualization Tools

This directory contains tools for analyzing and visualizing policies in the Networked Control System (NCS) simulator.

## Contents

- **`visualize_policy.py`**: Main visualization tool for plotting policy state evolution
- **`heuristic_policies.py`**: Collection of simple baseline policies
- **`__init__.py`**: Package initialization

## Visualization Tool

The `visualize_policy.py` script allows you to visualize how policies control the networked control system by plotting:
- 2D state space trajectories
- State magnitude evolution over time
- Transmission decisions
- Cumulative rewards
- Policy statistics comparison
- **Animated videos** showing state evolution in real-time (optional)

### Basic Usage

#### Visualize a Single Heuristic Policy

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policy always_send \
    --policy-type heuristic \
    --episode-length 500
```

#### Visualize a Trained SB3 Policy

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policy path/to/ppo_model.zip \
    --policy-type sb3 \
    --episode-length 500
```

#### Compare Multiple Policies

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policies always_send never_send send_every_5 path/to/ppo_model.zip \
    --policy-types heuristic heuristic heuristic sb3 \
    --labels "Always Send" "Never Send" "Send Every 5" "PPO" \
    --episode-length 500
```

#### Generate Animation Video

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policies always_send threshold_1.0 \
    --policy-types heuristic heuristic \
    --labels "Always Send" "Threshold 1.0" \
    --episode-length 300 \
    --generate-video \
    --video-format gif \
    --video-speedup 2
```

**Note**: For MP4 format, you need to install FFmpeg: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac). GIF format works without additional dependencies.

#### Trace Micro-slot Network Activity

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policy path/to/latest_model.pt \
    --policy-type marl_torch \
    --episode-length 500 \
    --network-trace \
    --trace-interval 50 \
    --trace-start 1
```

### Command-Line Arguments

**Required:**
- `--config`: Path to configuration JSON file

**Policy Selection (choose one):**
- `--policy`: Single policy to visualize (requires `--policy-type`)
- `--policies`: Multiple policies to compare (requires `--policy-types`)

**Policy Type:**
- `--policy-type`: Type of single policy (`sb3` or `heuristic`)
- `--policy-types`: Types of multiple policies (space-separated)

**Optional:**
- `--labels`: Custom labels for policies in plots
- `--episode-length`: Number of timesteps to simulate (default: 500)
- `--seed`: Random seed for reproducibility (default: 42)
- `--n-agents`: Optional override for agent count (default: read from checkpoint or config)
- `--show-estimates`: Show controller estimates in plots (default: True)
- `--output-dir`: Directory for saving visualizations (default: `vis/`)
- `--output-prefix`: Prefix for output filenames (default: timestamp)
- `--deterministic`: Force deterministic policy actions (default: False)
- `--network-trace`: Trace micro-slot network activity and render per-tick timelines
- `--trace-interval`: Tick interval for network traces (default: 50)
- `--trace-start`: First tick index to trace (default: 1)
- `--list-heuristics`: List available heuristic policies and exit

**Video/Animation:**
- `--generate-video`: Generate animated video of state evolution
- `--video-format`: Video format - `mp4` (requires FFmpeg) or `gif` (default: `mp4`)
- `--video-fps`: Frames per second for video (default: 30)
- `--video-speedup`: Speed multiplier (e.g., 2 = 2x speed, default: 1)

### Outputs

The tool generates the following files in the output directory:

**Always generated:**
1. **`{prefix}_state_evolution.png`**: 2D state space trajectory and magnitude plot
2. **`{prefix}_detailed_analysis.png`**: Comprehensive analysis including actions, rewards, and statistics
3. **`{prefix}_summary.json`**: Numerical summary statistics

**Generated when `--generate-video` is used:**
4. **`{prefix}_animation.mp4`** or **`{prefix}_animation.gif`**: Animated visualization showing:
   - Real-time state evolution in 2D space
   - State magnitude changing over time
   - Transmission decisions as they occur
   - Current timestep indicator

**Generated when `--network-trace` is used:**
5. **`{prefix}_{tag}_network_trace.jsonl`**: Micro-slot trace data (one JSON object per traced tick)
6. **`{prefix}_{tag}_network_tick_{tick}.png`**: Per-tick timeline plot of micro-slot activity

## Heuristic Policies

The `heuristic_policies.py` module provides simple baseline policies for comparison with learned policies.

### Available Policies

List all available policies:
```bash
python -m tools.visualize_policy --list-heuristics
```

#### Basic Policies

- **`always_send`**: Always transmit measurements at every timestep
- **`never_send`**: Never transmit measurements
- **`zero_wait`**: Transmit only when the previous packet is ACKed or dropped (no overlapping sends)

#### Periodic Policies

- **`send_every_2`**: Transmit every 2 timesteps
- **`send_every_5`**: Transmit every 5 timesteps
- **`send_every_10`**: Transmit every 10 timesteps

#### Stochastic Policies

- **`random_25`**: Transmit with 25% probability
- **`random_50`**: Transmit with 50% probability
- **`random_75`**: Transmit with 75% probability

#### Threshold-Based Policies

- **`threshold_0.5`**: Transmit when state magnitude ||x|| > 0.5
- **`threshold_1.0`**: Transmit when state magnitude ||x|| > 1.0
- **`threshold_2.0`**: Transmit when state magnitude ||x|| > 2.0

#### Adaptive Policies

- **`adaptive`**: Adaptive threshold based on state magnitude and channel throughput

### Using Heuristic Policies in Code

```python
from tools.heuristic_policies import get_heuristic_policy

# Get a policy by name
policy = get_heuristic_policy('always_send', n_agents=1)

# Use the policy
action, _ = policy.predict(observation, deterministic=True)
```

### Creating Custom Heuristic Policies

You can create custom policies by extending the `BaseHeuristicPolicy` class:

```python
from tools.heuristic_policies import BaseHeuristicPolicy
import numpy as np

class MyCustomPolicy(BaseHeuristicPolicy):
    def predict(self, observation, deterministic=True):
        # Your custom logic here
        action = 1 if some_condition else 0
        return action, None
```

## Examples

### Example 1: Compare Different Sending Frequencies

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policies send_every_2 send_every_5 send_every_10 \
    --policy-types heuristic heuristic heuristic \
    --labels "Every 2 steps" "Every 5 steps" "Every 10 steps" \
    --episode-length 300
```

### Example 2: Compare Threshold Policies

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policies threshold_0.5 threshold_1.0 threshold_2.0 always_send \
    --policy-types heuristic heuristic heuristic heuristic \
    --labels "Threshold 0.5" "Threshold 1.0" "Threshold 2.0" "Always Send" \
    --episode-length 300
```

### Example 3: Evaluate Trained PPO vs Baselines

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policies outputs/ppo_model.zip always_send adaptive \
    --policy-types sb3 heuristic heuristic \
    --labels "Trained PPO" "Always Send" "Adaptive" \
    --episode-length 500 \
    --output-prefix "ppo_comparison"
```

### Example 4: Custom Output Directory

```bash
python -m tools.visualize_policy \
    --config configs/perfect_comm_low_noise.json \
    --policy threshold_1.0 \
    --policy-type heuristic \
    --episode-length 1000 \
    --output-dir results/visualizations \
    --output-prefix "low_noise_threshold"
```

### Example 5: Generate Animation Video

```bash
# Generate GIF animation (no FFmpeg required)
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policies always_send adaptive \
    --policy-types heuristic heuristic \
    --labels "Always Send" "Adaptive Policy" \
    --episode-length 200 \
    --generate-video \
    --video-format gif \
    --video-fps 30 \
    --video-speedup 2
```

```bash
# Generate MP4 animation (requires FFmpeg)
python -m tools.visualize_policy \
    --config configs/perfect_comm.json \
    --policy outputs/ppo_model.zip \
    --policy-type sb3 \
    --episode-length 500 \
    --generate-video \
    --video-format mp4 \
    --video-fps 60 \
    --video-speedup 5
```

## Understanding the Plots

### State Evolution Plot

- **Left panel**: Shows the 2D state space trajectory
  - Solid lines: Actual plant states
  - Dashed lines: Controller's estimates (if `--show-estimates` is enabled)
  - Circle markers: Starting points
  - Square markers: Ending points
  - Red star: Target (origin)

- **Right panel**: State magnitude over time
  - Shows ||x|| (Euclidean norm of state vector)
  - Lower values indicate better control performance

### Detailed Analysis Plot

- **Top-left**: Transmission decisions (0 = no send, 1 = send)
- **Top-right**: Cumulative reward over time
- **Bottom-left**: Individual state dimensions over time
- **Bottom-right**: Bar chart comparing:
  - Total transmission count
  - Average reward
  - Final state error

### Animation Video

The animated video (when `--generate-video` is used) shows three synchronized panels:

- **Left panel**: 2D state space with moving points showing current position
  - Trajectories build up over time
  - Circle markers show current state position
  - Square markers show current estimate position (if enabled)
  - Timestep counter displayed in top-left

- **Middle panel**: State magnitude evolution
  - Line graph builds up over time showing ||x||
  - Shows how well the system is being controlled

- **Right panel**: Transmission decisions
  - Step plot showing when transmissions occur
  - Helps understand the policy's decision-making pattern

## Tips

1. **Choosing Episode Length**:
   - Short episodes (100-300): Quick comparison
   - Long episodes (500-1000): Better understanding of long-term behavior

2. **Comparing Policies**:
   - Compare at most 4-5 policies at once for readability
   - Use meaningful labels to distinguish policies

3. **Interpreting Results**:
   - Lower final state error = better stabilization
   - Higher cumulative reward = better overall performance
   - Transmission count = communication overhead

4. **Config Selection**:
   - Use `perfect_comm.json` to isolate policy behavior without network effects
   - Use default config to see policy performance with realistic network conditions

5. **Video Generation**:
   - Use GIF format if FFmpeg is not installed (works everywhere, larger file size)
   - Use MP4 format for smaller files and better quality (requires FFmpeg)
   - Adjust `--video-speedup` for longer episodes (2-5x recommended for episodes >200 steps)
   - Keep FPS at 30 for smooth playback, or increase to 60 for very detailed analysis
   - Shorter episodes (100-300 steps) work best for animations
   - Video generation can take 30-60 seconds depending on episode length and format

## Extending the Tool

### Adding New Visualizations

You can extend the visualization tool by adding new plotting functions. Follow the pattern in `plot_state_evolution_2d()` and `plot_detailed_analysis()`.

### Adding New Heuristic Policies

1. Create a new class in `heuristic_policies.py` extending `BaseHeuristicPolicy`
2. Implement the `predict()` method
3. Add it to the `HEURISTIC_POLICIES` dictionary
4. Test it with the visualization tool

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'stable_baselines3'`
- **Solution**: Install stable-baselines3: `pip install stable-baselines3`

**Issue**: `FileNotFoundError` when loading SB3 model
- **Solution**: Check the model path is correct and the `.zip` file exists

**Issue**: Plots look cluttered with many policies
- **Solution**: Reduce the number of policies or create separate comparisons

**Issue**: Episode ends early
- **Solution**: This is normal if the episode reaches the configured length. The warning can be safely ignored.

**Issue**: FFmpeg error when generating MP4 videos
- **Solution**: Install FFmpeg or use `--video-format gif` instead
  - Linux: `sudo apt-get install ffmpeg`
  - Mac: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html

**Issue**: Animation generation is slow
- **Solution**:
  - Use `--video-speedup` to skip frames (e.g., `--video-speedup 2` for 2x speed)
  - Reduce `--episode-length` for faster generation
  - Use GIF format which is faster than MP4

**Issue**: GIF file is too large
- **Solution**:
  - Use MP4 format instead (requires FFmpeg, much smaller files)
  - Increase `--video-speedup` to reduce total frames
  - Reduce `--episode-length`

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Gymnasium
- stable-baselines3 (only for loading SB3 policies)
- filterpy (for Kalman filter)
- FFmpeg (optional, only for MP4 video generation)

## License

Same as parent project.
