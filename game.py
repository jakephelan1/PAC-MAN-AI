import copy
import sys
from math import pi
from board import boards
import pygame
from player import Player
from ghost import Ghost
import random
import heapq

WIDTH = 720
HEIGHT = 755
FPS = 60
SPEED = 10
G_SPEED = 9

def dijkstra(graph, start, end):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    previous = {node: None for node in graph}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous[current_node]
            return path[::-1]

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return None

class game_info():
    def __init__(self, pacman, pacman_pos, g_coords, speed, pacman_direction, ghost_directions, pacman_moves, state, map, powerup_pos, points, points_g, graph, init_time, final_time):
        self.pacman = pacman
        self.pacman_pos = pacman_pos
        self.g_coords = g_coords
        self.speed = speed
        self.pacman_direction = pacman_direction
        self.ghost_directions = ghost_directions
        self.pacman_moves = pacman_moves
        self.state = state
        self.map = map
        self.powerup_pos = powerup_pos
        self.points = points
        self.points_g = points_g
        self.graph = graph
        self.init_time = init_time
        self.final_time = final_time
        self.current_time = pygame.time.get_ticks()

    def get_powerup_time_remaining(self):
        if self.pacman.powered_up:
            return max(0, (self.init_time + self.final_time) - self.current_time)
        return 0

    def get_powerup_time_remaining_normalized(self):
        if self.final_time > 0:
            return self.get_powerup_time_remaining() / self.final_time
        return 0


class pacman_game:
    def __init__(self, num_ghosts=0, power_pellets=False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.level = copy.deepcopy(boards[1]) if power_pellets else copy.deepcopy(boards[0])
        self.startBoard = copy.deepcopy(boards[1]) if power_pellets else copy.deepcopy(boards[0])
        self.graph = self.create_graph()
        self.timer = 121
        self.counter = 0
        self.flicker = False
        self.points = 0
        self.num1 = ((HEIGHT - 35) // 32)
        self.num2 = (WIDTH // 30)
        initial_x = 339
        initial_y = 520
        self.pacman = Player(initial_x, initial_y, pygame.transform.scale(pygame.image.load('Player/game.png'), (35, 35)))
        self.pressed_keys = set()
        self.lives = 3
        self.ghosts = []
        self.ghost_rects = []
        self.pacman_alive = True
        self.respawn_delay = 1000
        self.game_over = False
        self.direction_command = None
        self.respawn_time = 0
        self.initialize_ghosts(num_ghosts)
        self.init_time = -900000000
        self.final_time = 2000
        self.complete_lvl = False
        self.gPoints = 0
        self.info = None

    def initialize_ghosts(self, numGhosts=0):
        ghost_names = ["Pinky", "Blinky", "Inky", "Clyde"]
        self.ghosts = []

        for i in range(min(numGhosts, 4)):
            ghost = Ghost(
                random.randrange(12*self.num2, 17*self.num2),
                random.randrange(14*self.num1, 16*self.num1),
                ghost_names[i],
                pygame.transform.scale(pygame.image.load('Ghosts/0.png'), (30, 30))
            )
            self.ghosts.append(ghost)

    def draw_board(self):
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] == 1:
                    pygame.draw.circle(self.screen, 'white',
                                       (j * self.num2 + (.5 * self.num2), i * self.num1 + (.5 * self.num1)), 4)
                if self.level[i][j] == 2 and self.flicker:
                    pygame.draw.circle(self.screen, 'white',
                                       (j * self.num2 + (.5 * self.num2), i * self.num1 + (.5 * self.num1)), 10)
                if self.level[i][j] == 3:
                    pygame.draw.line(self.screen, 'blue', (j * self.num2 + (.5 * self.num2), i * self.num1),
                                     (j * self.num2 + (0.5 * self.num2), i * self.num1 + self.num1), 3)
                if self.level[i][j] == 4:
                    pygame.draw.line(self.screen, 'blue', (j * self.num2, i * self.num1 + (.5 * self.num1)),
                                     (j * self.num2 + self.num2, i * self.num1 + (.5 * self.num1)), 3)
                if self.level[i][j] == 5:
                    pygame.draw.arc(self.screen, 'blue',
                                    [(j * self.num2 - (.4 * self.num2) - 2), (i * self.num1 + (.5 * self.num1)), self.num2,
                                     self.num1], 0, pi / 2, 3)
                if self.level[i][j] == 6:
                    pygame.draw.arc(self.screen, 'blue',
                                    [(j * self.num2 + (.5 * self.num2)), (i * self.num1 + (.5 * self.num1)), self.num2, self.num1],
                                    pi / 2, pi, 3)
                if self.level[i][j] == 7:
                    pygame.draw.arc(self.screen, 'blue',
                                    [(j * self.num2 + (.5 * self.num2)), (i * self.num1 - (.4 * self.num1)), self.num2, self.num1],
                                    pi, (3 * pi) / 2, 3)
                if self.level[i][j] == 8:
                    pygame.draw.arc(self.screen, 'blue',
                                    [(j * self.num2 - (.4 * self.num2) - 1), (i * self.num1 - (.5 * self.num1)), self.num2,
                                     self.num1], (3 * pi) / 2, 2 * pi, 2)
                if self.level[i][j] == 9:
                    pygame.draw.line(self.screen, 'white', (j * self.num2, i * self.num1 + (.5 * self.num1)),
                                     (j * self.num2 + self.num2, i * self.num1 + (.5 * self.num1)), 3)

    def draw_player(self):
        pacman_rect = self.pacman.get_image().get_rect()
        pacman_rect.topleft = (self.pacman.get_x(), self.pacman.get_y())
        if self.pacman.get_direction() == 0 or self.pacman.get_direction() == -1:
            self.screen.blit(self.pacman.get_image(), pacman_rect)
        if self.pacman.get_direction() == 1:
            self.screen.blit(pygame.transform.flip(self.pacman.get_image(), True, False), pacman_rect)
        if self.pacman.get_direction() == 2:
            self.screen.blit(pygame.transform.rotate(self.pacman.get_image(), 90), pacman_rect)
        if self.pacman.get_direction() == 3:
            self.screen.blit(pygame.transform.rotate(self.pacman.get_image(), 270), pacman_rect)

    def draw_ghosts(self):
        for ghost in self.ghosts:
            ghost_rect = ghost.get_image().get_rect()
            ghost_rect.topleft = (ghost.get_x(), ghost.get_y())
            self.ghost_rects.append(ghost_rect)
        for i in range(len(self.ghosts)):
            self.screen.blit(self.ghosts[i].get_image(), self.ghost_rects[i])

    def in_tp_zone(self, ghost):
        node = ((ghost.get_x() + 16) // self.num2, (ghost.get_y() + 16) // self.num1)
        if (node[0] <= 1 or node[0] >= 27) and (node[1] == 14 or node[1] == 15):
            return True
        return False

    def check_available_moves(self):
        num3 = 20

        center_x = self.pacman.get_x() + 18
        center_y = self.pacman.get_y() + 18

        self.pacman.moves.clear()
        if 1 < center_x // self.num2 < 28:
            if self.level[(center_y - num3) // self.num1][center_x // self.num2] == 9:
                self.pacman.moves.add(2)
            if self.level[center_y // self.num1][(center_x - num3) // self.num2] < 3:
                self.pacman.moves.add(1)
            if self.level[center_y // self.num1][(center_x + num3) // self.num2] < 3:
                self.pacman.moves.add(0)
            if self.level[(center_y + num3) // self.num1][center_x // self.num2] < 3:
                self.pacman.moves.add(3)
            if self.level[(center_y - num3) // self.num1][center_x // self.num2] < 3:
                self.pacman.moves.add(2)

            if self.pacman.direction == 2 or self.pacman.direction == 3:
                if 14 < center_x % self.num2 < 16:
                    if self.level[(center_y + num3) // self.num1][center_x // self.num2] < 3:
                        self.pacman.moves.add(3)
                    if self.level[(center_y - num3) // self.num1][center_x // self.num2] < 3:
                        self.pacman.moves.add(2)
                if 14 < center_y % self.num1 < 16:
                    if self.level[center_y // self.num1][(center_x - self.num2) // self.num2] < 3:
                        self.pacman.moves.add(1)
                    if self.level[center_y // self.num1][(center_x + self.num2) // self.num2] < 3:
                        self.pacman.moves.add(0)

            if self.pacman.direction == 0 or self.pacman.direction == 1:
                if 14 < center_x % self.num2 < 16:
                    if self.level[(center_y + num3) // self.num1][center_x // self.num2] < 3:
                        self.pacman.moves.add(3)
                    if self.level[(center_y - num3) // self.num1][center_x // self.num2] < 3:
                        self.pacman.moves.add(2)
                if 14 < center_y % self.num1 < 16:
                    if self.level[center_y // self.num1][(center_x - num3) // self.num2] < 3:
                        self.pacman.moves.add(1)
                    if self.level[center_y // self.num1][(center_x + num3) // self.num2] < 3:
                        self.pacman.moves.add(0)
        else:
            self.pacman.moves.add(0)
            self.pacman.moves.add(1)

    def check_available_moves_g(self, obj, x, y):
        walls = self.wall_barrier(obj, x, y)
        for i in range(4):
            if i not in walls:
                obj.moves.add(i)
            elif i in obj.moves:
                obj.moves.remove(i)

    def point_collection(self, x, y):
        if x // 30 < 23 and self.pacman_alive:
            try:
                if self.level[y // self.num1][x // self.num2] == 1:
                    self.level[y // self.num1][x // self.num2] = -1
                    self.points += 1
                    if not any(1 in row for row in self.level):
                        self.complete_lvl = True
                if self.level[y // self.num1][x // self.num2] == 2:
                    self.level[y // self.num1][x // self.num2] = -1
                    self.init_time = pygame.time.get_ticks()
            except:
                pass

    def wall_barrier(self, obj, x, y):
        barrier = set()
        if isinstance(obj, Player):
            n = 9
            sp = SPEED
        else:
            sp = G_SPEED
            n = 8
        if x // 30 < 23:
            num3 = n + sp
            if self.level[(y) // self.num1][(x + num3) // self.num2] >= 3:
                barrier.add(0)
            elif 0 in barrier:
                barrier.remove(0)
            if self.level[(y) // self.num1][(x - num3) // self.num2] >= 3:
                barrier.add(1)
            elif 1 in barrier:
                barrier.remove(1)
            if self.level[(y - num3) // self.num1][x // self.num2] >= 3:
                barrier.add(2)
            elif 2 in barrier:
                barrier.remove(2)
            if self.level[(y + num3) // self.num1][x // self.num2] >= 3:
                barrier.add(3)
            elif 3 in barrier:
                barrier.remove(3)
        return barrier

    def collision_detector(self):
        global pacman_alive, respawn_time
        if not self.ghosts:
            return
        if pygame.time.get_ticks() - self.init_time <= self.final_time:
            self.pacman.set_state(False)
            self.pacman.set_state(True)
            self.pacman.set_image(pygame.transform.scale(pygame.image.load('Player/games.png'), (30, 30)))
        else:
            self.pacman.set_image(pygame.transform.scale(pygame.image.load('Player/game.png'), (30, 30)))
            self.pacman.set_state(False)
        collide = False
        g = None
        for i in range(len(self.ghost_rects)):
            if self.ghost_rects[i].colliderect(self.pacman_rect):
                collide = True
                g = self.ghosts[i]
                break
        if collide and self.pacman_alive and not self.pacman.get_state():
            self.pressed_keys.clear()
            self.pacman_alive = False
            self.gPoints += 200
            for ghost in self.ghosts:
                ghost.set_direction(-1)
            self.respawn_time = pygame.time.get_ticks()
        elif collide and self.pacman_alive and self.pacman.get_state():
            pygame.time.wait(500)
            self.final_time += 500
            self.points += 50
            self.gPoints -= 50
            g.set_x(random.randrange(12*self.num2, 17*self.num2))
            g.set_y(random.randrange(14*self.num2, (16*self.num2) - 30))
            g.out_of_box = False

    def blink_points(self):
        if self.counter < 20:
            self.counter += 1
            self.flicker = False
        else:
            self.counter = 0
            while self.counter < 7:
                self.counter += 1
            self.flicker = True

    def handle_events(self, run):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                self.pressed_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self.pressed_keys.discard(event.key)

        if pygame.K_RIGHT in self.pressed_keys and 0 in self.pacman.moves:
            self.pacman.set_direction(0)
        elif pygame.K_LEFT in self.pressed_keys and 1 in self.pacman.moves:
            self.pacman.set_direction(1)
        elif pygame.K_UP in self.pressed_keys and 2 in self.pacman.moves:
            self.pacman.set_direction(2)
        elif pygame.K_DOWN in self.pressed_keys and 3 in self.pacman.moves:
            self.pacman.set_direction(3)

        return run

    def move_pacman(self):
        center_x = self.pacman.get_x() + 18
        center_y = self.pacman.get_y() + 18

        self.check_available_moves()

        if center_x - 18 <= 0 or center_x + 18 >= 720:
            if center_x - 18 <= 0 and self.pacman.get_direction() == 1:
                self.pacman.set_x(720)
            elif center_x + 18 >= 720 and self.pacman.get_direction() == 0:
                self.pacman.set_x(0)

        if self.pacman.get_direction() == 0 and 0 in self.pacman.moves:
            self.pacman.set_x(self.pacman.get_x() + SPEED)
        elif self.pacman.get_direction() == 1 and 1 in self.pacman.moves:
            self.pacman.set_x(self.pacman.get_x() - SPEED)
        elif self.pacman.get_direction() == 2 and 2 in self.pacman.moves:
            self.pacman.set_y(self.pacman.get_y() - SPEED)
        elif self.pacman.get_direction() == 3 and 3 in self.pacman.moves:
            self.pacman.set_y(self.pacman.get_y() + SPEED)

    def move_ghosts(self):
        positions = {}
        for ghost in self.ghosts:
            current_pos = ((ghost.get_x() + 16) // self.num2, (ghost.get_y() + 16) // self.num1)
            self.check_available_moves_g(ghost, ghost.get_x() + 16, ghost.get_y() + 16)

            if ghost.direction_cooldown <= 0 and ghost.at_intersection():
                if self.pacman.powered_up:
                    path = self.get_shortest_path(ghost, fleeing=True)
                else:
                    if ghost.name == "Clyde":
                        pacman_pos = ((self.pacman.get_x() + 18) // self.num2, (self.pacman.get_y() + 18) // self.num1)
                        distance = abs(current_pos[0] - pacman_pos[0]) + abs(current_pos[1] - pacman_pos[1])
                        ghost.in_corner = current_pos == (2, 30)

                        if ghost.going_to_corner == False and ghost.in_corner == False:
                            if distance <= 8 and not ghost.in_corner:
                                ghost.going_to_corner = True
                            else:
                                ghost.going_to_corner = False

                        elif ghost.in_corner:
                            ghost.going_to_corner = False

                        if ghost.going_to_corner:
                            path = self.get_shortest_path(ghost, end=(2, 30))
                        else:
                            path = self.get_shortest_path(ghost)

                    elif ghost.name == "Blinky":
                        path = self.get_shortest_path(ghost)
                    elif ghost.name == "Pinky":
                        path = self.get_shortest_path(ghost, track_ahead=True)
                    elif ghost.name == "Inky":
                        path = self.get_inky_path(ghost)

                if path and len(path) > 1:
                    nextPos = path[1]
                    dx = nextPos[0] - current_pos[0]
                    dy = nextPos[1] - current_pos[1]

                    if dx > 0 and 0 in ghost.moves:
                        ghost.target_direction = 0
                    elif dx < 0 and 1 in ghost.moves:
                        ghost.target_direction = 1
                    elif dy < 0 and 2 in ghost.moves:
                        ghost.target_direction = 2
                    elif dy > 0 and 3 in ghost.moves:
                        ghost.target_direction = 3
                    else:
                        ghost.target_direction = random.choice(list(ghost.moves))

                    if not self.in_tp_zone(ghost):
                        if ghost.target_direction != ghost.get_direction():
                            ghost.set_direction(ghost.target_direction)
                            ghost.direction_cooldown = ghost.direction_cooldown_time
                            ghost.persistence_time = 30
            else:
                ghost.direction_cooldown -= 1
                ghost.persistence_time -= 1

            if not self.in_tp_zone(ghost):
                if ghost.persistence_time <= 0 and random.random() < 0.1:
                    new_direction = random.choice(list(ghost.moves - {ghost.get_direction()}))
                    ghost.set_direction(new_direction)
                    ghost.persistence_time = 30

                if ghost.get_direction() not in ghost.moves and ghost.out_of_box:
                    new_direction = random.choice(list(ghost.moves))
                    ghost.set_direction(new_direction)

            positions[ghost] = current_pos

        collisions = set()
        for ghost1, pos1 in positions.items():
            for ghost2, pos2 in positions.items():
                if ghost1 != ghost2 and pos1 == pos2:
                    collisions.add((ghost1, ghost2))

        handled_ghosts = set()
        for ghost1, ghost2 in collisions:
            if ghost1 in handled_ghosts or ghost2 in handled_ghosts:
                break
            if ghost1.out_of_box and ghost2.out_of_box:
                num = random.choice([1, 2])

                if num == 1:
                    ghost1.set_direction(self.get_opposite_direction(ghost2.get_direction()))
                    handled_ghosts.add(ghost1)
                else:
                    ghost2.set_direction(self.get_opposite_direction(ghost1.get_direction()))
                    handled_ghosts.add(ghost2)

    def get_inky_path(self, ghost):
        blinky = self.ghosts[1]
        blinkyPos = ((blinky.get_x() + 16) // self.num2, (blinky.get_y() + 16) // self.num1)
        pacman_pos = ((self.pacman.get_x() + 18) // self.num2, (self.pacman.get_y() + 18) // self.num1)

        dx = pacman_pos[0] - blinkyPos[0]
        dy = pacman_pos[1] - blinkyPos[1]
        targetPos = (pacman_pos[0] + dx, pacman_pos[1] + dy)

        if targetPos not in self.graph:
            targetPos = min(self.graph.keys(), key=lambda x: abs(x[0] - targetPos[0]) + abs(x[1] - targetPos[1]))

        return self.get_shortest_path(ghost, end=targetPos)

    def get_opposite_direction(self, direction):
        if direction == 0:
            return 1
        elif direction == 1:
            return 0
        elif direction == 2:
            return 3
        elif direction == 3:
            return 2

    def create_graph(self):
        graph = {}
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] < 3:
                    node = (j, i)
                    graph[node] = {}
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(self.level) and 0 <= nj < len(self.level[i]) and self.level[ni][nj] < 3:
                            graph[node][(nj, ni)] = 1

                    if j == 0:
                        graph[node][(len(self.level[i])-1, i)] = 1
                    elif j == len(self.level[i])-1:
                        graph[node][(0, i)] = 1
        return graph

    def check_for_clear_shot(self, ghost):
        start = ((ghost.get_x() + 16) // self.num2, (ghost.get_y() + 16) // self.num1)
        pacman_pos=((self.pacman.get_x() + 18) // self.num2, (self.pacman.get_y() + 18) // self.num1)

        if start[0] == pacman_pos[0]:
            min_y, max_y = min(start[1], pacman_pos[1]), max(start[1], pacman_pos[1])
            return all((start[0], y) in self.graph for y in range(min_y, max_y + 1))
        elif start[1] == pacman_pos[1]:
            min_x, max_x = min(start[0], pacman_pos[0]), max(start[0], pacman_pos[0])
            return all((x, start[1]) in self.graph for x in range(min_x, max_x + 1))

    def get_shortest_path(self, ghost, fleeing=False, track_ahead=False, end=None):
        start = (((ghost.get_x() + 16) // self.num2) % 30, (ghost.get_y() + 16) // self.num1)
        pacman_pos = ((self.pacman.get_x() + 18) // self.num2, (self.pacman.get_y() + 18) // self.num1)

        if end != None:
            return dijkstra(self.graph, start, end)

        if start not in self.graph:
            start = min(self.graph.keys(), key=lambda x: abs(x[0] - start[0]) + abs(x[1] - start[1]))

        if not track_ahead:
            if fleeing:
                escape_points = self.find_escape_points(ghost, pacman_pos)
                if not escape_points:
                    return None
                end = random.choice(escape_points)

            else:
                if pacman_pos not in self.graph:
                    pacman_pos = min(self.graph.keys(), key=lambda x: abs(x[0] - pacman_pos[0]) + abs(x[1] - pacman_pos[1]))
                end = pacman_pos
        else:
            same_lane = self.check_for_clear_shot(ghost)
            dir = self.pacman.get_direction()
            if not same_lane:
                if pacman_pos in self.graph:
                    if dir == 0:
                        end = max(self.graph[pacman_pos], key=lambda x: x[0])
                    elif dir == 1:
                        end = min(self.graph[pacman_pos], key=lambda x: x[0])
                    elif dir == 2:
                        end = min(self.graph[pacman_pos], key=lambda x: x[1])
                    elif dir == 3:
                        end = max(self.graph[pacman_pos], key=lambda x: x[1])
                else:
                    nearest_pos = min(self.graph.keys(), key=lambda x: abs(x[0] - pacman_pos[0]) + abs(x[1] - pacman_pos[1]))
                    end = nearest_pos
            else:
                end = pacman_pos

        if end not in self.graph:
            end = min(self.graph.keys(), key=lambda x: abs(x[0] - end[0]) + abs(x[1] - end[1]))

        path = dijkstra(self.graph, start, end)

        if path is None or len(path) < 2:
            neighbors = list(self.graph[start].keys())
            return [start, random.choice(neighbors)] if neighbors else None

        return path

    def find_escape_points(self, ghost, pacman_pos, numPoints=4):
        escape_points = []
        ghostPos = ((ghost.get_x() + 16) // self.num2, (ghost.get_y() + 16) // self.num1)

        ghost_pacman_dist = ((ghostPos[0] - pacman_pos[0]) ** 2 + (ghostPos[1] - pacman_pos[1]) ** 2) ** 0.5
        for node in self.graph.keys():
            node_pacman_dist = ((node[0] - pacman_pos[0]) ** 2 + (node[1] - pacman_pos[1]) ** 2) ** 0.5
            if node_pacman_dist > ghost_pacman_dist:
                escape_points.append((node, node_pacman_dist))

        escape_points.sort(key=lambda x: x[1], reverse=True)
        return [point[0] for point in escape_points[:numPoints]]

    def handle_death(self):
        if not self.pacman_alive and pygame.time.get_ticks() - self.respawn_time >= self.respawn_delay:
            self.lives -= 1
            self.pacman_alive = True
            self.pacman.set_x(339)
            self.pacman.set_y(520)
            self.direction_command = None
            self.pacman.set_direction(0)
            self.respawn_time = 0
            if self.lives < 1:
                self.game_over = True
            for ghost in self.ghosts:
                ghost.out_of_box = False
                ghost.set_x(random.randrange(12*self.num2, 17*self.num2))
                ghost.set_y(random.randrange(14*self.num2, (16*self.num2) - 30))
                ghost.set_y(random.randrange(14*self.num2, (16*self.num2) - 30))

    def handle_ghosts(self):
        if not self.ghosts:
            return {}
        g_coords = {}
        for ghost in self.ghosts:
            ghost_rect = ghost.get_image().get_rect()
            ghost_rect.topleft = (ghost.get_x(), ghost.get_y())
            self.ghost_rects.append(ghost_rect)

            x = ghost.get_x() + 16
            y = ghost.get_y() + 16

            if not ghost.out_of_box:
                target_x = 15 * self.num2
                target_y = 13 * self.num1

                if abs(x - target_x) > G_SPEED:
                    ghost.set_x(ghost.get_x() + (G_SPEED if x < target_x else -G_SPEED))
                elif y > target_y:
                    ghost.set_y(ghost.get_y() - G_SPEED)
                else:
                    ghost.out_of_box = True
            else:
                near_tp = x <= G_SPEED * 2 or x >= WIDTH - G_SPEED * 2 - 32

                if x <= 0 and ghost.get_direction() == 1:
                    ghost.set_x(WIDTH - 32)
                elif x >= WIDTH - 32 and ghost.get_direction() == 0:
                    ghost.set_x(0)
                elif not near_tp and ghost.get_direction() not in ghost.moves:
                    new_direction = random.choice(list(ghost.moves))
                    ghost.set_direction(new_direction)

                if ghost.get_direction() == 0:
                    ghost.set_x(ghost.get_x() + G_SPEED)
                elif ghost.get_direction() == 1:
                    ghost.set_x(ghost.get_x() - G_SPEED)
                elif ghost.get_direction() == 2:
                    ghost.set_y(ghost.get_y() - G_SPEED)
                elif ghost.get_direction() == 3:
                    ghost.set_y(ghost.get_y() + G_SPEED)

            g_coords[ghost] = (ghost.get_x() + 16, ghost.get_y() + 16)

        return g_coords


    def get_ghost_dirs(self):
        if not self.ghosts:
            return {}
        ghost_dirs = {}
        for ghost in self.ghosts:
            ghost_dirs[ghost] = ghost.get_direction()
        return ghost_dirs

    def get_powerup_pos(self):
        powerup_pos = []
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] == 2:
                    powerup_pos.append((j * self.num2, i * self.num1))
        return powerup_pos

    def draw(self):
        font = pygame.font.Font('freesansbold.ttf', 15)
        font2 = pygame.font.Font('freesansbold.ttf', 70)
        font3 = pygame.font.Font('freesansbold.ttf', 20)
        self.screen.fill((0, 0, 0))
        self.draw_board()
        self.blink_points()
        if self.pacman_alive:
            self.draw_player()
        self.draw_ghosts()
        label = font.render("Points: " + str(self.points), 1, 'white')
        lives_label = font.render("Lives: " + str(self.lives), 1, 'white')
        self.screen.blit(label, (10, 735))
        self.screen.blit(lives_label, (WIDTH - 80, 735))


    def loop(self):
        center_point = (self.pacman.get_x() + 18, self.pacman.get_y() + 18)
        self.point_collection(center_point[0], center_point[1])
        self.pacman_rect = self.pacman.get_image().get_rect()
        self.pacman_rect.topleft = (self.pacman.get_x(), self.pacman.get_y())
        self.ghost_rects = []

        if self.ghosts:
            self.move_ghosts()
            g_coords = self.handle_ghosts()
            self.collision_detector()
        else:
            g_coords = {}

        self.handle_death()
        self.move_pacman()

        ghost_directions = self.get_ghost_dirs() if self.ghosts else {}
        powerup_pos = self.get_powerup_pos()

        info = game_info(self.pacman, center_point, g_coords, SPEED,
                             self.pacman.get_direction(), ghost_directions,
                             self.pacman.moves, self.pacman.get_state(),
                             self.level, powerup_pos, self.points, self.gPoints,
                             self.graph, self.init_time, self.final_time)

        return info

def main():
    game = pacman_game(4)
    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(FPS)
        game_info = game.loop()
        run = game.handle_events(run)
        game.draw()
        pygame.display.update()
        if game.game_over or game.complete_lvl:
            run = False
            break

if __name__ == "__main__":
    main()



