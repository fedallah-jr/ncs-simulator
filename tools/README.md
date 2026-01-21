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

#### Visualize a Heuristic Policy (Multi-Agent)

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policy always_send \
    --policy-type heuristic \
    --episode-length 500
```

#### Visualize a Trained MARL Policy

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policy path/to/latest_model.pt \
    --policy-type marl_torch \
    --episode-length 500 \
    --generate-video --per-agent-videos
```

#### Visualize an OpenAI-ES Policy

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policy path/to/latest_model.npz \
    --policy-type es \
    --episode-length 500
```

#### Compare Multiple Policies

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policies always_send never_send send_every_5 path/to/latest_model.pt \
    --policy-types heuristic heuristic heuristic marl_torch \
    --labels "Always Send" "Never Send" "Send Every 5" "MARL" \
    --episode-length 500
```

#### Generate Animation Video

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policies always_send threshold_1.0 \
    --policy-types heuristic heuristic \
    --labels "Always Send" "Threshold 1.0" \
    --episode-length 300 \
    --generate-video \
    --video-speedup 2 \
    --per-agent-videos
```
**Note**: Generating MP4s requires FFmpeg (`sudo apt-get install ffmpeg` on Linux, `brew install ffmpeg` on Mac).

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
- `--policy-type`: Type of single policy (`marl_torch`, `es`, `openai_es`, or `heuristic`)
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
- `--video-fps`: Frames per second for video (default: 30)
- `--video-speedup`: Speed multiplier (e.g., 2 = 2x speed, default: 1)

### Outputs

The tool generates the following files in the output directory:

**Always generated:**
1. **`{prefix}_marl_summary.png`**: Mean state error and mean action rate summary
2. **`{prefix}_{tag}_actions.png`**: Per-agent action raster for each policy
3. **`{prefix}_{tag}_state_space.png`**: Combined multi-agent 2D state space (when `state_dim >= 2`)
4. **`{prefix}_summary.json`**: Numerical summary statistics

**Generated when `--generate-video` is used:**
5. **`{prefix}_{tag}_animation.mp4`**: Animated visualization of all agents
6. **`{prefix}_{tag}_agent_{i}.mp4`**: Optional per-agent videos when `--per-agent-videos` is set

**Generated when `--network-trace` is used:**
7. **`{prefix}_{tag}_network_trace.jsonl`**: Micro-slot trace data (one JSON object per traced tick)
8. **`{prefix}_{tag}_network_tick_{tick}.png`**: Per-tick timeline plot of micro-slot activity

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
from tools.visualize_policy import MultiAgentHeuristicPolicy

# Create a multi-agent heuristic wrapper
policy = MultiAgentHeuristicPolicy("always_send", n_agents=4, seed=0, deterministic=True)

# Use the policy
actions = policy.act(obs_dict)
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
    --config configs/marl_mixed_plants.json \
    --policies send_every_2 send_every_5 send_every_10 \
    --policy-types heuristic heuristic heuristic \
    --labels "Every 2 steps" "Every 5 steps" "Every 10 steps" \
    --episode-length 300
```

### Example 2: Compare Threshold Policies

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policies threshold_0.5 threshold_1.0 threshold_2.0 always_send \
    --policy-types heuristic heuristic heuristic heuristic \
    --labels "Threshold 0.5" "Threshold 1.0" "Threshold 2.0" "Always Send" \
    --episode-length 300
```

### Example 3: Evaluate Trained MARL vs Baselines

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policies outputs/latest_model.pt always_send adaptive \
    --policy-types marl_torch heuristic heuristic \
    --labels "Trained MARL" "Always Send" "Adaptive" \
    --episode-length 500 \
    --output-prefix "marl_comparison"
```

### Example 4: Custom Output Directory

```bash
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policy threshold_1.0 \
    --policy-type heuristic \
    --episode-length 1000 \
    --output-dir results/visualizations \
    --output-prefix "low_noise_threshold"
```

### Example 5: Generate Animation Video

```bash
# Generate MP4 animation (requires FFmpeg)
python -m tools.visualize_policy \
    --config configs/marl_mixed_plants.json \
    --policy outputs/latest_model.pt \
    --policy-type marl_torch \
    --episode-length 200 \
    --generate-video \
    --video-fps 60 \
    --video-speedup 5
```

## Understanding the Plots

### MARL Summary Plot

- **Top panel**: Mean state error across agents over time
- **Bottom panel**: Mean action rate across agents

### Action Raster

- Rows correspond to agents, columns to timesteps
- Filled cells indicate transmission actions (1)

### State Space Plot

- 2D trajectories for all agents (and estimates if `--show-estimates` is enabled)
- Useful for spotting coordination patterns and divergence between agents

### Animation Video

- Animated 2D state space showing all agents simultaneously
- Optional per-agent videos via `--per-agent-videos`

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
   - Use `configs/marl_mixed_plants.json` to evaluate with representative multi-agent setups
   - If a config does not define `system.n_agents`, pass `--n-agents` explicitly

5. **Video Generation**:
   - Install FFmpeg for MP4 generation
   - Adjust `--video-speedup` for longer episodes (2-5x recommended for episodes >200 steps)
   - Keep FPS at 30 for smooth playback, or increase to 60 for very detailed analysis
   - Shorter episodes (100-300 steps) work best for animations
   - Video generation can take 30-60 seconds depending on episode length and format

## Extending the Tool

### Adding New Visualizations

You can extend the visualization tool by adding new plotting functions. Follow the pattern in `plot_marl_comparison_summary()` and `plot_marl_action_raster()`.

### Adding New Heuristic Policies

1. Create a new class in `heuristic_policies.py` extending `BaseHeuristicPolicy`
2. Implement the `predict()` method
3. Add it to the `HEURISTIC_POLICIES` dictionary
4. Test it with the visualization tool

## Behavioral Cloning Dataset

Generate a behavioral cloning dataset from a heuristic policy (default: `zero_wait`):
```bash
python -m tools.generate_bc_dataset --config configs/marl_mixed_plants.json \
  --episodes 50 --episode-length 500 --output outputs/bc_zero_wait.npz
```

The saved `.npz` contains step-level `obs`, `actions`, `rewards`, and `dones` plus metadata
such as `n_agents` and `obs_dim`, which can be used to pretrain MAPPO with `--bc-dataset`.

## Troubleshooting

**Issue**: Plots look cluttered with many policies
- **Solution**: Reduce the number of policies or create separate comparisons

**Issue**: Episode ends early
- **Solution**: This is normal if the episode reaches the configured length. The warning can be safely ignored.

**Issue**: FFmpeg error when generating MP4 videos
- **Solution**: Install FFmpeg and retry
  - Linux: `sudo apt-get install ffmpeg`
  - Mac: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html

**Issue**: Animation generation is slow
- **Solution**:
  - Use `--video-speedup` to skip frames (e.g., `--video-speedup 2` for 2x speed)
  - Reduce `--episode-length` for faster generation

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Gymnasium
- torch (for marl_torch checkpoints)
- jax/jaxlib/flax (for ES checkpoints)
- filterpy (for Kalman filter)
- FFmpeg (optional, only for MP4 video generation)

## License

Same as parent project.
