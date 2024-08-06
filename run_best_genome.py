import os
import pickle
import neat
from game import pacman_game
from AI import AI
import pygame

def load_genome(filename):
    with open(filename, 'rb') as f:
        genome, config = pickle.load(f)
    return genome, config

def run_game_with_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = pacman_game(num_ghosts=4, power_pellets=True)
    ai_tester = AI()

    final_score, steps, lives = ai_tester.run_pacman(net, game)

    pygame.quit()
    print(f"Final Score: {final_score}")
    print(f"Steps taken: {steps}")
    print(f"Lives remaining: {lives}")

genome, config = load_genome("overall_best_genome.pkl")
run_game_with_genome(genome, config)

