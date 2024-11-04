import pygame
import random
import math
from typing import List, Tuple
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)


class Resource:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.collected = False
        self.size = 5

    def draw(self, screen):
        if not self.collected:
            pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), self.size)


class Agent:
    def __init__(self, x: float, y: float, team: str):
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.team = team  # 'collector' or 'opposer'
        self.speed = 2
        self.size = 8
        self.carrying_resource = None
        self.perception_radius = 100

    def distance_to(self, other_x: float, other_y: float) -> float:
        return math.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

    def move_towards(self, target_x: float, target_y: float):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > 0:
            self.velocity_x = (dx / distance) * self.speed
            self.velocity_y = (dy / distance) * self.speed

    def apply_swarm_rules(self, agents: List['Agent']):
        # Parameters for swarm behavior
        separation_radius = 20
        cohesion_radius = 50
        alignment_radius = 40

        separate_x = separate_y = 0
        cohesion_x = cohesion_y = 0
        align_x = align_y = 0

        neighbors = 0

        for other in agents:
            if other != self and other.team == self.team:
                distance = self.distance_to(other.x, other.y)

                # Separation
                if distance < separation_radius:
                    separate_x += self.x - other.x
                    separate_y += self.y - other.y

                # Cohesion
                if distance < cohesion_radius:
                    cohesion_x += other.x
                    cohesion_y += other.y
                    neighbors += 1

                # Alignment
                if distance < alignment_radius:
                    align_x += other.velocity_x
                    align_y += other.velocity_y

        if neighbors > 0:
            cohesion_x /= neighbors
            cohesion_y /= neighbors
            align_x /= neighbors
            align_y /= neighbors

            self.velocity_x += (separate_x * 0.3 + (cohesion_x - self.x) * 0.1 + align_x * 0.2)
            self.velocity_y += (separate_y * 0.3 + (cohesion_y - self.y) * 0.1 + align_y * 0.2)

            # Normalize velocity
            speed = math.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
            if speed > self.speed:
                self.velocity_x = (self.velocity_x / speed) * self.speed
                self.velocity_y = (self.velocity_y / speed) * self.speed

    def update(self, resources: List[Resource], agents: List['Agent'], base_x: float, base_y: float):
        if self.team == 'collector':
            if self.carrying_resource:
                # Return to base with resource
                self.move_towards(base_x, base_y)
            else:
                # Look for nearest uncollected resource
                nearest_resource = None
                min_distance = float('inf')

                for resource in resources:
                    if not resource.collected:
                        distance = self.distance_to(resource.x, resource.y)
                        if distance < min_distance and distance < self.perception_radius:
                            min_distance = distance
                            nearest_resource = resource

                if nearest_resource:
                    self.move_towards(nearest_resource.x, nearest_resource.y)

        elif self.team == 'opposer':
            # Look for nearest collector carrying a resource
            nearest_target = None
            min_distance = float('inf')

            for agent in agents:
                if agent.team == 'collector' and agent.carrying_resource:
                    distance = self.distance_to(agent.x, agent.y)
                    if distance < min_distance and distance < self.perception_radius:
                        min_distance = distance
                        nearest_target = agent

            if nearest_target:
                self.move_towards(nearest_target.x, nearest_target.y)

        # Apply swarm behavior rules
        self.apply_swarm_rules(agents)

        # Update position
        self.x += self.velocity_x
        self.y += self.velocity_y

        # Keep within bounds
        self.x = max(0, min(self.x, WINDOW_WIDTH))
        self.y = max(0, min(self.y, WINDOW_HEIGHT))

    def draw(self, screen):
        color = BLUE if self.team == 'collector' else RED
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)
        if self.carrying_resource:
            pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), 3)


class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Swarm Resource Collection Simulation")
        self.clock = pygame.time.Clock()

        # Initialize agents
        self.collectors = [Agent(random.randint(0, WINDOW_WIDTH),
                                 random.randint(0, WINDOW_HEIGHT),
                                 'collector') for _ in range(10)]
        self.opposers = [Agent(random.randint(0, WINDOW_WIDTH),
                               random.randint(0, WINDOW_HEIGHT),
                               'opposer') for _ in range(5)]
        self.agents = self.collectors + self.opposers

        # Initialize resources
        self.resources = [Resource(random.randint(0, WINDOW_WIDTH),
                                   random.randint(0, WINDOW_HEIGHT)) for _ in range(15)]

        # Base location for collectors
        self.base_x = WINDOW_WIDTH // 2
        self.base_y = WINDOW_HEIGHT - 30

        self.collected_resources = 0
        self.stolen_resources = 0

    def handle_collisions(self):
        # Check collector-resource collisions
        for collector in self.collectors:
            if not collector.carrying_resource:
                for resource in self.resources:
                    if not resource.collected and collector.distance_to(resource.x, resource.y) < 10:
                        resource.collected = True
                        collector.carrying_resource = resource
                        break

        # Check opposer-collector collisions
        for opposer in self.opposers:
            for collector in self.collectors:
                if (collector.carrying_resource and
                        opposer.distance_to(collector.x, collector.y) < 15):
                    collector.carrying_resource.collected = False
                    collector.carrying_resource = None
                    self.stolen_resources += 1

        # Check collector-base collisions
        for collector in self.collectors:
            if (collector.carrying_resource and
                    collector.distance_to(self.base_x, self.base_y) < 20):
                collector.carrying_resource = None
                self.collected_resources += 1

    def run(self):
        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update
            for agent in self.agents:
                agent.update(self.resources, self.agents, self.base_x, self.base_y)

            self.handle_collisions()

            # Draw
            self.screen.fill(BLACK)

            # Draw base
            pygame.draw.rect(self.screen, WHITE,
                             (self.base_x - 20, self.base_y - 10, 40, 20))

            # Draw resources
            for resource in self.resources:
                resource.draw(self.screen)

            # Draw agents
            for agent in self.agents:
                agent.draw(self.screen)

            # Draw scores
            font = pygame.font.Font(None, 36)
            collected_text = font.render(f'Collected: {self.collected_resources}', True, WHITE)
            stolen_text = font.render(f'Stolen: {self.stolen_resources}', True, WHITE)
            self.screen.blit(collected_text, (10, 10))
            self.screen.blit(stolen_text, (10, 50))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()