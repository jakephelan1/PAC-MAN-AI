# NEAT Pac-Man AI
Created by: [Jake Phelan](https://github.com/jakephelan1)

https://github.com/user-attachments/assets/ab39c9c8-bb98-434a-a55d-c94d40151798

## Project Overview

This project implements an AI agent that learns to play Pac-Man using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The AI evolves neural networks to control Pac-Man, aiming to maximize the score while avoiding ghosts and collecting pellets.

### Features

- Custom Pac-Man game implementation
- NEAT algorithm for evolving neural network controllers
- Visualizations of network structures and training progress
- Configurable ghost behaviors and game mechanics
- Support for power pellets and ghost vulnerability states
- Ability to save and load trained models

### Tools Used

- Python: Primary programming language
- Pygame: Game rendering and user input handling
- NEAT-Python: Implementation of the NEAT algorithm
- Matplotlib: Plotting training statistics

## Setup and Installation

1. Clone the Repository
   ```
   git clone https://github.com/jakephelan1/PAC-MAN-AI
   cd PAC-MAN-AI
   ```

2. Install Dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run the Training Process
   ```
   python AI.py
   ```

4. Run the Best Trained Model
   ```
   python run_best_genome.py
   ```

## Project Structure

- `AI.py`: Main script for training the NEAT algorithm
- `game.py`: Pac-Man game logic and rendering
- `board.py`: Board for the game with and without power pellets
- `visualize.py`: Functions for visualizing networks and statistics
- `ghost.py`: Ghost class implementation
- `player.py`: Pac-Man player class implementation
- `run_best_genome.py`: Script to run the best trained genome
- `config.txt`: NEAT algorithm configuration file

## Training Process

The AI uses the NEAT algorithm to evolve a population of neural networks. Each network controls Pac-Man's movements based on game state inputs. The fitness of each genome is determined by the score achieved in the game, with bonuses for eating ghosts and penalties for losing lives.

The training process gradually increases the difficulty by adding more ghosts and eventually introducing power pellets. This staged approach allows the AI to learn basic movement and pellet collection before tackling more complex strategies.

## Usage

As of now, `overall_best_genome.pkl` is saved in the repo and you can watch the best-performing AI play Pac-Man by running `run_best_genome.py`. The game window will open, showing the AI-controlled Pac-Man navigating the maze, avoiding ghosts, and collecting pellets. If you want to retrain, run `AI.py` and then `run_best_genome.py`

## Customization

You can modify the NEAT algorithm parameters in `config.txt` to experiment with different evolution settings. The game mechanics and ghost behaviors can be adjusted in `game.py` to create new challenges for the AI. Also modify the fitness function as the training progresses to reward different behaviors in different training steps.

## Acknowledgments

- The NEAT algorithm by Kenneth O. Stanley and Risto Miikkulainen
- The Pygame community for the game development framework
- Code Bullet for inspiration
