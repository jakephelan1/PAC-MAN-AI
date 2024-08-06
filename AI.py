import copy
import pickle
import os
from functools import partial
from game import dijkstra
import math
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import neat
import os
from game import pacman_game
import multiprocessing
import random
import visualize


WIDTH = 720
HEIGHT = 755
FPS = 60
BOARD = 0
num1 = ((HEIGHT - 35) // 32)
num2 = (WIDTH // 30)

class PositionHistory:
    def __init__(self, size=30, height=32, width=30, decay_factor=0.95):
        self.history = np.zeros((height, width))
        self.size = size
        self.positions = []
        self.decay_factor = decay_factor

    def add_position(self, position):
        self.positions.append(position)
        self.history[position[1], position[0]] += 1
        if len(self.positions) > self.size:
            old_pos = self.positions.pop(0)
            self.history[old_pos[1], old_pos[0]] -= self.decay_factor

    def get_position_score(self, position):
        return self.history[position[1], position[0]]

class AI:
    def __init__(self):
        self.game = pacman_game()

    @staticmethod
    def direction_to_vector(direction):
        return {
            0: (1, 0),
            1: (-1, 0),
            2: (0, -1),
            3: (0, 1)
        }[direction]

    @staticmethod
    def get_next_position(current_pos, direction):
        dx, dy = AI.direction_to_vector(direction)
        return (current_pos[0] + dx, current_pos[1] + dy)

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def calculate_move_scores(current_position, valid_moves, position_history, game_info):
        move_scores = {}
        for move in valid_moves:
            next_pos = AI.get_next_position(current_position, move)
            next_pos = (next_pos[0] % 30, next_pos[1] % 32)

            history_score = position_history.get_position_score(next_pos)

            pellet_score = 5 if game_info.map[next_pos[1]][next_pos[0]] == 1 else 0
            power_pellet_score = 20 if game_info.map[next_pos[1]][next_pos[0]] == 2 else 0

            ghost_score = 0
            if game_info.g_coords:
                if game_info.get_powerup_time_remaining_normalized() > 0.2:
                    ghost_score = max(32 - AI.manhattan_distance(next_pos, (ghost_pos[0] // num2, ghost_pos[1] // num1))
                                      for ghost_pos in game_info.g_coords.values())
                else:
                    ghost_score = min(AI.manhattan_distance(next_pos, (ghost_pos[0] // num2, ghost_pos[1] // num1))
                                      for ghost_pos in game_info.g_coords.values())

            if game_info.get_powerup_time_remaining_normalized() > 0.2:
                move_scores[move] = (2 * pellet_score) + power_pellet_score + (3 * ghost_score) - (3 * history_score)
            else:
                move_scores[move] = pellet_score + power_pellet_score + ghost_score - (5 * history_score)

        return move_scores

    @staticmethod
    def choose_move(net_output, valid_moves, current_position, position_history, game_info, recent_moves):
        valid_outputs = {move: net_output[move] for move in valid_moves}
        max_output = max(valid_outputs.values())
        if max_output > 0:
            normalized_net_output = {move: value / max_output for move, value in valid_outputs.items()}
        else:
            normalized_net_output = {move: 1.0 / len(valid_moves) for move in valid_moves}

        move_scores = AI.calculate_move_scores(current_position, valid_moves, position_history, game_info)
        max_score = max(move_scores.values())
        if max_score > 0:
            normalized_move_scores = {move: score / max_score for move, score in move_scores.items()}
        else:
            normalized_move_scores = {move: 1.0 / len(valid_moves) for move in valid_moves}

        combined_scores = {move: (0.75 * normalized_net_output[move] + 0.25 * normalized_move_scores[move])
                           for move in valid_moves}

        if len(recent_moves) >= 3:
            last_three_moves = recent_moves[-3:]
            if last_three_moves[0] == last_three_moves[2] and last_three_moves[1] == AI.get_opposite_direction(last_three_moves[0]):
                combined_scores[last_three_moves[1]] = -1000

        return max(combined_scores, key=combined_scores.get)

    @staticmethod
    def get_opposite_direction(direction):
        return {0: 1, 1: 0, 2: 3, 3: 2}.get(direction, None)

    def train_pacman(self, genome, config, num_ghosts, power_pellets):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = pacman_game(power_pellets=power_pellets)
        game.initialize_ghosts(num_ghosts)
        run = True
        clock = pygame.time.Clock()
        points = 0
        steps = 0
        base_max_steps = 600
        max_steps = base_max_steps + (100 * num_ghosts)
        if power_pellets:
            max_steps += 200
        last_score_change = 0
        last_score = 0
        empty_space_penalty = 0
        initial_lives = game.lives
        life_loss_penalty = 100
        lives = 3
        ghost_avoidance_reward = 0
        previous_ghost_distances = {}
        visited_positions = []
        recent_moves = []
        position_history = PositionHistory()
        cleared_points = False

        direction_change_cooldown = 0
        COOLDOWN_DURATION = 3

        while run and steps < max_steps:
            clock.tick(60)
            game_info = game.loop()
            run = game.handle_events(run)
            game.draw()
            pygame.display.update()
            if game.game_over or game.complete_lvl:
                cleared_points = game.complete_lvl
                run = False
                break
            steps += 1

            if steps - last_score_change > 300:
                break

            if game.points > last_score:
                last_score = game.points
                last_score_change = steps

            if steps % 100 == 0:
                max_steps += 10

            pacman_x = int(game_info.pacman_pos[0] // num2) % 30
            pacman_y = int(game_info.pacman_pos[1] // num1)
            current_position = (pacman_x, pacman_y)

            if 0 <= pacman_y < len(game.level) and 0 <= pacman_x < len(game.level[0]):
                if game.level[pacman_y][pacman_x] == -1:
                    empty_space_penalty += 0.04

            if direction_change_cooldown == 0 or game.pacman.get_direction() not in game_info.pacman_moves:
                if game.pacman.at_intersection():
                    inputs = self.prepare_inputs(game_info, max_ghosts=4)
                    output = net.activate(inputs)
                    assert all(0 <= o <= 1 for o in output), f"Network output not normalized: {output}"
                    valid_moves = [dir for dir in range(4) if dir in game_info.pacman_moves]
                    chosen_dir = self.choose_move(output, valid_moves, current_position, position_history, game_info, recent_moves)

                    if chosen_dir != game.pacman.get_direction():
                        game.pacman.set_direction(chosen_dir)
                        recent_moves.append(chosen_dir)
                    if len(recent_moves) > 10:
                        recent_moves.pop(0)
                    direction_change_cooldown = COOLDOWN_DURATION

            if direction_change_cooldown > 0:
                direction_change_cooldown -= 1

            position_history.add_position(current_position)

            current_ghost_distances = {}
            for ghost, pos in game_info.g_coords.items():
                distance = ((pos[0] - game_info.pacman_pos[0])**2 + (pos[1] - game_info.pacman_pos[1])**2)**0.5
                current_ghost_distances[ghost] = distance

                if ghost in previous_ghost_distances:
                    if distance > previous_ghost_distances[ghost] and not ghost.dead:
                        ghost_avoidance_reward += 0.5

            previous_ghost_distances = current_ghost_distances

            points = game.points

        lives_lost = initial_lives - game.lives
        life_penalty = lives_lost * life_loss_penalty

        base_fitness = points
        ghost_factor = 1 + (num_ghosts * 0.1)
        diversity_bonus = len(set(visited_positions)) * 0.1

        if cleared_points:
            fitness = 4000 + (2000 * game.lives)

        # Don't add ghost_avoidance_reward after power pellets are added!
        final_fitness = fitness if cleared_points else (base_fitness * ghost_factor) + diversity_bonus - empty_space_penalty - life_penalty

        return max(final_fitness, 0.01)

    def prepare_inputs(self, game_info, max_ghosts=4):
        inputs = []

        nearest_pellet_dir = self.find_nearest_pellet(game_info)
        inputs.extend(nearest_pellet_dir)

        power_pellet_dir = self.find_nearest_power_pellet_direction(game_info)
        inputs.extend(power_pellet_dir)

        info = self.calculate_ghost_info(game_info)
        inputs.extend(info)

        current_direction = game_info.pacman.get_direction()
        direction_input = [0, 0, 0, 0]
        if 0 <= current_direction < 4:
            direction_input[current_direction] = 1
        inputs.extend(direction_input)

        inputs.append(game_info.get_powerup_time_remaining_normalized())

        return inputs

    def isValidPos(self, x, y, game_info):
        return 0 <= x < len(game_info.map[0]) and 0 <= y < len(game_info.map)

    def find_nearest_pellet(self, game_info):
        pacman_x, pacman_y = game_info.pacman_pos[0] // num2, game_info.pacman_pos[1] // num1
        min_distance = float('inf')
        direction = [0, 0, 0, 0]
        width = len(game_info.map[0])
        height = len(game_info.map)
        closest_pellet = None

        if not self.isValidPos(pacman_x, pacman_y, game_info):
            return direction

        for y, row in enumerate(game_info.map):
            for x, cell in enumerate(row):
                if cell == 1:
                    dx = x - pacman_x
                    dy = y - pacman_y
                    if pacman_y in [13, 14, 15] and abs(dx) > width // 2:
                        dx = width - abs(dx) if dx > 0 else -(width - abs(dx))
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_pellet = (x, y)

        if closest_pellet:
            dx = closest_pellet[0] - pacman_x
            dy = closest_pellet[1] - pacman_y
            if pacman_y in [13, 14, 15] and abs(dx) > width // 2:
                dx = width - abs(dx) if dx > 0 else -(width - abs(dx))
            total = abs(dx) + abs(dy)
            if total > 0:
                direction = [
                    max(0, dx / total) if dx > 0 else 0,
                    max(0, -dx / total) if dx < 0 else 0,
                    max(0, -dy / total) if dy < 0 else 0,
                    max(0, dy / total) if dy > 0 else 0
                ]

        return direction

    def find_nearest_power_pellet_direction(self, game_info):
        pacman_x, pacman_y = game_info.pacman_pos[0] // num2, game_info.pacman_pos[1] // num1
        direction = [0, 0, 0, 0]
        width = len(game_info.map[0])
        min_distance = float('inf')
        closest_power_pellet = None

        if not self.isValidPos(pacman_x, pacman_y, game_info):
            return direction

        for y, row in enumerate(game_info.map):
            for x, cell in enumerate(row):
                if cell == 2:
                    dx = x - pacman_x
                    dy = y - pacman_y
                    if pacman_y in [13, 14, 15] and abs(dx) > width // 2:
                        dx = width - abs(dx) if dx > 0 else -(width - abs(dx))
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_power_pellet = (x, y)

        if closest_power_pellet:
            dx = closest_power_pellet[0] - pacman_x
            dy = closest_power_pellet[1] - pacman_y
            if pacman_y in [13, 14, 15] and abs(dx) > width // 2:
                dx = width - abs(dx) if dx > 0 else -(width - abs(dx))
            total = abs(dx) + abs(dy)
            if total > 0:
                direction = [
                    max(0, dx / total) if dx > 0 else 0,
                    max(0, -dx / total) if dx < 0 else 0,
                    max(0, -dy / total) if dy < 0 else 0,
                    max(0, dy / total) if dy > 0 else 0
                ]

        return direction

    def calculate_ghost_info(self, game_info):
        pacman_x, pacman_y = (game_info.pacman_pos[0] // num2) % 30, game_info.pacman_pos[1] // num1
        pacman_pos = (pacman_x, pacman_y)

        if pacman_pos not in game_info.graph:
            print(f"Warning: Pacman position {pacman_pos} not in graph")
            return [0, 0, 0, 0]

        ghost_data = [0, 0, 0, 0]
        max_distance = 15
        scaled_ghost_positions = []

        for ghost_pos in game_info.g_coords.values():
            ghost_x, ghost_y = (ghost_pos[0] // num2) % 30, ghost_pos[1] // num1
            scaled_ghost_pos = (ghost_x, ghost_y)
            scaled_ghost_positions.append(scaled_ghost_pos)

            dx = min((ghost_x - pacman_x) % 30, (pacman_x - ghost_x) % 30)
            dy = ghost_y - pacman_y

            if dx != 0:
                data = max(0, (max_distance - dx) / max_distance)
                ghost_data[1 if ghost_x > pacman_x else 0] += data
            if dy != 0:
                data = max(0, (max_distance - abs(dy)) / max_distance)
                ghost_data[2 if dy > 0 else 3] += data

        max_dir = max(ghost_data)
        if max_dir > 0:
            ghost_data = [d / max_dir for d in ghost_data]

        try:
            farthest_point = max(game_info.graph.keys(),
                                 key=lambda x: min(self.manhattan_distance(x, ghost_pos) for ghost_pos in scaled_ghost_positions))
            path = dijkstra(game_info.graph, pacman_pos, farthest_point)

            if path and len(path) >= 2:
                next_pos = path[1]
                dx = next_pos[0] - pacman_pos[0]
                dy = next_pos[1] - pacman_pos[1]

                path_direction = [0, 0, 0, 0]
                if dx > 0:
                    path_direction[0] = 1
                elif dx < 0:
                    path_direction[1] = 1
                elif dy < 0:
                    path_direction[2] = 1
                elif dy > 0:
                    path_direction[3] = 1
            else:
                path_direction = [0, 0, 0, 0]
        except Exception as e:
            path_direction = [0, 0, 0, 0]

        combined_info = [0.7 * g + 0.3 * p for g, p in zip(ghost_data, path_direction)]

        max_value = max(combined_info)
        if max_value > 0:
            combined_info = [c / max_value for c in combined_info]
        return combined_info

    def run_pacman(self, net, game, max_steps=None):
        run = True
        clock = pygame.time.Clock()
        steps = 0
        recent_moves = []
        position_history = PositionHistory()

        direction_change_cooldown = 0
        COOLDOWN_DURATION = 3

        while run and (max_steps is None or steps < max_steps):
            clock.tick(60)
            game_info = game.loop()
            run = game.handle_events(run)
            game.draw()
            pygame.display.update()
            if game.game_over or game.complete_lvl:
                run = False
                break
            steps += 1

            pacman_x = int(game_info.pacman_pos[0] // num2) % 30
            pacman_y = int(game_info.pacman_pos[1] // num1)
            current_position = (pacman_x, pacman_y)

            if direction_change_cooldown == 0 or game.pacman.get_direction() not in game_info.pacman_moves:
                if game.pacman.at_intersection():
                    inputs = self.prepare_inputs(game_info, max_ghosts=4)
                    output = net.activate(inputs)
                    assert all(0 <= o <= 1 for o in output), f"Network output not normalized: {output}"
                    valid_moves = [dir for dir in range(4) if dir in game_info.pacman_moves]
                    chosen_dir = self.choose_move(output, valid_moves, current_position, position_history, game_info, recent_moves)

                    if chosen_dir != game.pacman.get_direction():
                        game.pacman.set_direction(chosen_dir)
                        recent_moves.append(chosen_dir)
                    if len(recent_moves) > 10:
                        recent_moves.pop(0)
                    direction_change_cooldown = COOLDOWN_DURATION

            if direction_change_cooldown > 0:
                direction_change_cooldown -= 1

            position_history.add_position(current_position)

        return game.points, steps, game.lives

def evaluate_genome_safe(genome, config, num_ghosts, power_pellets):
    try:
        simulation = AI()
        fitness = simulation.train_pacman(genome, config, num_ghosts, power_pellets)
        return genome.key, max(fitness, 0.01)
    except Exception as e:
        print(f"Error evaluating genome {genome.key}: {str(e)}")
        print(f"Error occurred in file {e.__traceback__.tb_frame.f_code.co_filename}, line {e.__traceback__.tb_lineno}")
        return genome.key, 0.01

def eval_genomes(genomes, config, num_ghosts, power_pellets):
    with multiprocessing.Pool() as pool:
        eval_function = partial(evaluate_genome_safe, config=config, num_ghosts=num_ghosts, power_pellets=power_pellets)
        results = pool.map(eval_function, [genome for _, genome in genomes])

    for (genome_id, genome), (key, fitness) in zip(genomes, results):
        if genome.key == key:
            genome.fitness = fitness
        else:
            print(f"Warning: Genome key mismatch. Expected {genome.key}, got {key}")


def run_neat(config, checkpoint=None):
    os.makedirs('models', exist_ok=True)
    node_names = {
        -1: 'Pellet Right', -2: 'Pellet Left', -3: 'Pellet Up', -4: 'Pellet Down',
        -5: 'Power Pellet Right', -6: 'Power Pellet Left', -7: 'Power Pellet Up', -8: 'Power Pellet Down',
        -9: 'Ghost Info Right', -10: 'Ghost Info Left', -11: 'Ghost Info Up', -12: 'Ghost Info Down',
        -13: 'PAC-MAN Right', -14: 'PAC-MAN Left', -15: 'PAC-MAN Up', -16: 'PACM-MAN Down',
        -17: 'Power-Up Time',
        0: 'Move Right', 1: 'Move Left', 2: 'Move Up', 3: 'Move Down'
    }
    if checkpoint:
        print(f"Restoring from checkpoint: {checkpoint}")
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Starting a new evolution")
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    max_ghosts = 4
    generations_per_ghost = 75
    total_generations = 450

    def eval_genomes_wrapper(genomes, config):
        num_ghosts = min(max_ghosts, p.generation // generations_per_ghost)
        power_pellets = p.generation >= 375

        if p.generation % generations_per_ghost == 0:
            if p.generation < 375:
                print(f"Increasing to {num_ghosts} ghosts")
            else:
                print('Adding Power Pellets')

        eval_genomes(genomes, config, num_ghosts, power_pellets)

        best_genome = max(genomes, key=lambda x: x[1].fitness)[1]
        if p.generation % 10 == 0 and p.generation != 0:
            visualize.draw_net(config, best_genome, view=False,
                               filename=f'Models/network_gen_{p.generation}',
                               node_names=node_names)

    winner = p.run(eval_genomes_wrapper, 71)

    print(f"Best overall fitness: {winner.fitness}")

    with open("overall_best_genome.pkl", "wb") as f:
        pickle.dump((winner, config), f)

    visualize.draw_net(config, winner, True, node_names=node_names, filename='Models/winner_network')

    return winner

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    checkpoint = 'Checkpoints/neat-checkpoint-380'

    run_neat(config, checkpoint)