# 🧩 Rubik's Cube Neural Solver

AI-powered Rubik's cube solver using **Genetic Algorithm** to evolve **Neural Network** weights.

![Architecture](https://img.shields.io/badge/Python-PyTorch-blue) ![Frontend](https://img.shields.io/badge/TypeScript-Three.js-green)

## 📁 Project Structure

```
CUBE/
├── python/                    # Training backend
│   ├── cube/                  # Cube logic
│   │   ├── cube_state.py     # State representation & moves
│   │   └── cube_env.py       # Training environment
│   ├── neural/               # Neural network
│   │   ├── network.py        # PyTorch model
│   │   └── weight_export.py  # Export for TensorFlow.js
│   ├── genetic/              # Genetic algorithm
│   │   ├── individual.py     # Individual representation
│   │   ├── fitness.py        # Fitness evaluation
│   │   └── evolution.py      # Evolution engine
│   ├── train.py              # Main training script
│   └── requirements.txt
│
└── frontend/                  # Visualization
    ├── src/
    │   ├── cube/             # Cube logic (TS)
    │   ├── visualization/    # Three.js renderer
    │   ├── neural/           # TensorFlow.js inference
    │   └── main.ts           # Entry point
    ├── index.html
    └── package.json
```

## 🚀 Quick Start

### 1. Train the Model (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Train with default settings
python train.py --population 50 --generations 100 --scramble-depth 5

# Train with more generations
python train.py --population 100 --generations 500 --scramble-depth 10
```

**Training Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population` | 50 | Number of individuals |
| `--generations` | 100 | Number of generations |
| `--scramble-depth` | 5 | Moves to scramble cubes |
| `--max-steps` | 30 | Max moves to solve |
| `--mutation-rate` | 0.1 | Mutation probability |
| `--output` | weights | Output directory |

### 2. Run the Frontend (TypeScript)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open `http://localhost:5173` in your browser.

### 3. Load Trained Weights

1. After training, find `best_weights.json` in `python/weights/run_YYYYMMDD_HHMMSS/`
2. In the web app, click **"📂 Load Weights"**
3. Select the JSON file
4. Click **"🔀 Scramble"** then **"✨ Solve"**

## 🧠 How It Works

### Architecture

```
Input (324) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(18, Softmax) → Move
```

- **Input**: One-hot encoded cube state (54 stickers × 6 colors)
- **Output**: Probability distribution over 18 possible moves

### Genetic Algorithm

1. **Population**: Each individual = neural network weights
2. **Fitness**: Average reward over N scrambled cubes
3. **Selection**: Tournament selection
4. **Crossover**: Uniform crossover of weights
5. **Mutation**: Gaussian noise on weights
6. **Elitism**: Keep top performers

### Training Tips

- Start with low `scramble-depth` (3-5) for faster initial learning
- Gradually increase difficulty as the model improves
- Use `--load checkpoint.json` to resume training

## 📊 Expected Results

| Scramble Depth | Generations | Solve Rate |
|----------------|-------------|------------|
| 3 | 50 | ~80% |
| 5 | 100 | ~60% |
| 10 | 500+ | ~30% |

> ⚠️ Genetic algorithms are stochastic. Results may vary between runs.

## 🎮 Controls

| Button | Action |
|--------|--------|
| 🔀 Scramble | Random scramble |
| ↩️ Reset | Return to solved state |
| ✨ Solve | Use neural network to solve |
| Manual moves | R, R', L, L', U, U', etc. |

## 📝 License

MIT
