# Rocket League Bot

Reinforcement learning bot for Rocket League using [RLGym](https://rlgym.org/) and [rlgym-ppo](https://github.com/AechPro/rlgym-ppo). Training runs in [RocketSim](https://github.com/ZealanL/RocketSim) (no game window); you can then play against the trained policy via [RLBot](https://rlbot.org/). The project is based on the [RLBot Python example](https://github.com/RLBot/python-interface/wiki).

---

## Prerequisites

- **Python 3.11** (3.12+ is not supported by rlgym without patches)
- **Rocket League** installed (Steam or Epic)
- **RLBotServer.exe** (see below) – required for running matches
- **RLBot** and the game for playing against the bot (not required for training only)

---

## RLBotServer.exe

To run matches (`python run.py`), RLBot needs **RLBotServer.exe** in the project root. You can either download a release or build it from source.

### Option A: Download a release

1. Go to [RLBot/core](https://github.com/RLBot/core) and open the **Releases** page.
2. Download **RLBotServer.exe** (or the appropriate build for your OS) from the latest release.
3. Place **RLBotServer.exe** in the **root directory** of this project (same folder as `run.py`).

### Option B: Build from source (RLBot Core)

If you want to build RLBotServer.exe yourself (e.g. to get the latest changes):

1. **Install .NET 8 SDK**  
   [https://dotnet.microsoft.com/en-us/download/dotnet/8.0](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)

2. **Install an IDE** (one of):
   - Visual Studio 2022 (used for initial Core development)
   - Rider
   - VS Code

3. **Clone the Core repo and submodules:**
   ```bash
   git clone https://github.com/RLBot/core.git
   cd core
   git submodule update --init
   ```

4. **Build:**
   - **Visual Studio 2022:** Open the solution, build in **Release** mode. Compiled binaries are at `RLBotCS\bin\Release\net8.0`.
   - **Command line:** From the repo root, run:
     ```bash
     dotnet build -c Release
     ```
     Output is in `RLBotCS\bin\Release\net8.0`.

5. Copy the built **RLBotServer.exe** (and any required DLLs if not bundled) into this project’s root directory.

For more details (formatting, flatbuffers, deployment), see the [RLBot Core README](https://github.com/RLBot/core).

---

## Quick start (run a match)

1. Install Python 3.11 or later.
2. Create and activate a virtual environment:
   - Windows: `py -3.11 -m venv venv` then `.\venv\Scripts\Activate.ps1`
   - Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure **RLBotServer.exe** is in the project root (see above).
5. Set **Steam vs Epic** in **`rlbot.toml`**: `launcher = "Steam"` or `launcher = "Epic"` under `[rlbot]`.
6. Start a match: `python run.py`

You can also use **`dev.toml`** for development (it has slightly different match settings).

---

## Environment setup (detailed)

### 1. Create a virtual environment and install dependencies

From the project root:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want **GPU (CUDA)** for faster training, install PyTorch with CUDA **after** the above:

```powershell
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

### 2. Set Rocket League launcher (Steam vs Epic)

RLBot needs to know which version of Rocket League you have. Edit **`rlbot.toml`** at the top:

```toml
[rlbot]
# "Steam", "Epic", "Custom", "NoLaunch"
launcher = "Epic"
```

- Use **`launcher = "Epic"`** if you have Rocket League on the Epic Games Store.
- Use **`launcher = "Steam"`** if you have it on Steam.

Save the file. If this is wrong, RLBot will not start the game when you run a match.

---

## Running the training loop

Training uses RocketSim only (no Rocket League window). Activate your venv, then from the project root:

**Quick test (~1–2 minutes)** – verifies the pipeline and saves one checkpoint:

```powershell
python train.py --quick-test
```

**Full training** – runs until 1 billion timesteps (or you stop it). Checkpoints save every 1M steps under `data/checkpoints/`:

```powershell
python train.py
```

You can reduce `n_proc` in `train.py` (e.g. to 8 or 16) if you have fewer CPU cores. Press **`c`** to save a checkpoint, **`q`** to checkpoint and quit.

---

## Updating the reward function

Rewards are defined in **`src/training/rewards.py`** and wired in **`train.py`**.

### Custom reward classes

In **`src/training/rewards.py`** you’ll find (and can add) classes that implement `RewardFunction`:

- **`reset(agents, initial_state, shared_info)`** – called when an episode starts.
- **`get_rewards(agents, state, is_terminated, is_truncated, shared_info)`** – returns a `Dict[AgentID, float]` of rewards per agent.

Example: the existing `SpeedTowardBallReward` rewards moving toward the ball; `InAirReward` rewards being in the air; `VelocityBallToGoalReward` rewards ball velocity toward the opponent’s goal.

### Wiring rewards in the training script

In **`train.py`**, the environment uses a **`CombinedReward`** that sums several rewards with weights. Find the `reward_fn` block in `build_rlgym_v2_env()` (around lines 75–80):

```python
reward_fn = CombinedReward(
    (InAirReward(), 0.002),
    (SpeedTowardBallReward(), 0.01),
    (VelocityBallToGoalReward(), 0.1),
    (GoalReward(), 10.0),
)
```

To change behavior:

1. **Add a new reward:** implement a class in `src/training/rewards.py`, import it in `train.py`, and add a tuple to `CombinedReward`, e.g. `(MyNewReward(), 0.05)`.
2. **Change weights:** adjust the second number in each tuple (higher = that reward matters more).
3. **Remove a reward:** delete or comment out the corresponding line in `CombinedReward`.

After editing, run `python train.py` (or `--quick-test`) as usual; no separate build step.

---

## Playing against the bot

After training (or after a quick test), you can run a 1v1 match: you (Human, Blue) vs the PPO bot (Orange).

### 1. Find the checkpoint folder

Checkpoints are saved under:

```
data/checkpoints/rlgym-ppo-run-<timestamp>/<timesteps>/
```

For example: `data/checkpoints/rlgym-ppo-run-1234567890/25000/` or `.../1000000/`. The folder must contain **`PPO_POLICY.pt`**.

### 2. Set the checkpoint path

Set the **`PPO_CHECKPOINT`** environment variable to that folder (the one that contains `PPO_POLICY.pt`).

**PowerShell:**

```powershell
$env:PPO_CHECKPOINT = "C:\Users\PC\...\Rocket_League_Bot\data\checkpoints\rlgym-ppo-run-<TIMESTAMP>\25000"
```

Or from the project root with a relative path:

```powershell
cd C:\Users\PC\PersonalCodingProjects\Rocket_League_Bots\Rocket_League_Bot
$env:PPO_CHECKPOINT = "data\checkpoints\rlgym-ppo-run-<TIMESTAMP>\25000"
```

Replace `<TIMESTAMP>` and `25000` with your actual run and timestep folder.

### 3. Start the match

With the same venv activated and **`rlbot.toml`** using the correct **`launcher`** (Steam or Epic):

```powershell
python run.py
```

RLBot will start Rocket League (if needed) and the match. You play as Blue; the PPO bot is Orange.

To use the hand-coded example bot instead of the PPO bot, edit **`rlbot.toml`** and change the Orange car’s `config_file` from `"src/ppo_bot.toml"` to `"src/bot.toml"`.

---

## Changing the bot

- **Bot behavior** is controlled by:
  - **`src/bot.py`** – hand-coded example bot (chase ball, flips, etc.)
  - **`src/ppo_bot.py`** – PPO-trained policy (set `PPO_CHECKPOINT` to use it)
- **Bot appearance** (car, decal, colors, etc.) is controlled by **`src/loadout.toml`**.

Switch which bot is used in a match by editing the `config_file` for that car in **`rlbot.toml`** (e.g. `src/bot.toml` vs `src/ppo_bot.toml`).

---

## Configuring for the v5 botpack

To package your bot for the RLBot v5 botpack:

1. Install PyInstaller: `pip install pyinstaller`
2. Build a single-file executable:
   ```bash
   pyinstaller --onefile src/bot.py --paths src
   ```
   This creates **`bot.spec`** in the project root.
3. Create **`bob.toml`** in the same directory as the spec file (project root) with:
   ```toml
   [[config]]
   project_name = "RocketLeagueBot"
   bot_configs = ["src/bot.toml"]

   [config.builder_config]
   builder_type = "pyinstaller"
   entry_file = "bot.spec"
   ```
   - **`project_name`** – name of your bot’s folder in the botpack  
   - **`bot_configs`** – list of bot configs to include (add `"src/ppo_bot.toml"` if you want the PPO bot in the pack)  
   - **`entry_file`** – the spec file name (rename `bot.spec` if you like, and update this)
4. Commit **`bot.spec`** and **`bob.toml`** to your repo. **`bob.toml`** cannot be renamed.

---

## Summary

| Goal | Command / Action |
|------|-------------------|
| RLBotServer.exe | Download from [RLBot/core](https://github.com/RLBot/core) releases, or build with .NET 8 + Visual Studio (see above) |
| Setup | `py -3.11 -m venv venv`, activate, `pip install -r requirements.txt` |
| Steam/Epic | Set `launcher = "Steam"` or `"Epic"` in `rlbot.toml` |
| Run a match | `python run.py` (exe must be in project root) |
| Quick test | `python train.py --quick-test` |
| Full training | `python train.py` |
| Change rewards | Edit `src/training/rewards.py` and `CombinedReward` in `train.py` |
| Change bot behavior | Edit `src/bot.py` or use PPO bot via `src/ppo_bot.toml` + `PPO_CHECKPOINT` |
| Play vs PPO bot | Set `PPO_CHECKPOINT` to checkpoint folder, then `python run.py` |
