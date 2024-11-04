from __future__ import annotations
import pygame
import random
import math
from typing import List, Tuple
import numpy as np
from enum import Enum


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
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)


class ResourceType(Enum):
    LIGHT = 1  # Single agent can carry
    MEDIUM = 2  # Two agents needed
    HEAVY = 3  # Three agents needed


class Resource:
    def __init__(self, x: float, y: float, resource_type: ResourceType):
        self.x = x
        self.y = y
        self.collected = False
        self.type = resource_type
        self.carriers = []  # List of agents carrying this resource
        self.size = 5 + (resource_type.value * 2)  # Size increases with weight
        self.value = resource_type.value * 10  # Heavier resources are worth more points

    def get_color(self):
        if self.type == ResourceType.LIGHT:
            return GREEN
        elif self.type == ResourceType.MEDIUM:
            return YELLOW
        else:
            return PURPLE

    def draw(self, screen):
        if not self.collected:
            pygame.draw.circle(screen, self.get_color(), (int(self.x), int(self.y)), self.size)


class Agent:
    def __init__(self, x: float, y: float, team: str):
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.team = team
        self.speed = 2
        self.size = 8
        self.carrying_resource = None
        self.perception_radius = 150
        self.communication_radius = 200
        self.target = None
        self.role = 'searching'  # 'searching', 'gathering', 'returning'
        self.energy = 100
        self.waiting_time = 0
        self.collaborators = []
        self.last_position = (x, y)
        self.stuck_time = 0

    def distance_to(self, other_x: float, other_y: float) -> float:
        return math.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

    def move_towards(self, target_x: float, target_y: float, slow_down_radius=50):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > 0:
            speed_factor = min(1.0, distance / slow_down_radius) if distance < slow_down_radius else 1.0
            self.velocity_x = (dx / distance) * self.speed * speed_factor
            self.velocity_y = (dy / distance) * self.speed * speed_factor

    def evaluate_resource(self, resource: Resource, agents: List['Agent']) -> float:
        if resource.collected:
            return 0

        distance = self.distance_to(resource.x, resource.y)
        nearby_agents = sum(1 for agent in agents
                            if agent.team == self.team and
                            agent.distance_to(resource.x, resource.y) < self.communication_radius and
                            not agent.carrying_resource)  # Only count available agents

        # Calculate value based on distance, resource value, and available help
        value = resource.value / (1 + distance * 0.01)

        # Heavily reduce value if we don't have enough agents nearby
        if resource.type.value > nearby_agents:
            value *= 0.2

        return value

    def should_separate_from_group(self, agents: List['Agent']) -> bool:
        if self.role != 'searching':
            return False

        nearby_idle_agents = sum(1 for agent in agents
                                 if agent.team == self.team and
                                 agent != self and
                                 agent.role == 'searching' and
                                 self.distance_to(agent.x, agent.y) < 50)
        return nearby_idle_agents > 3  # Separate if too many idle agents nearby

    def get_separation_vector(self, agents: List['Agent']) -> Tuple[float, float]:
        sep_x = sep_y = 0
        count = 0

        for agent in agents:
            if (agent != self and agent.team == self.team and
                    not agent.carrying_resource):
                distance = self.distance_to(agent.x, agent.y)
                if distance < 50:  # Separation radius
                    factor = 1.0 - (distance / 50.0)  # Stronger separation when closer
                    sep_x += (self.x - agent.x) * factor
                    sep_y += (self.y - agent.y) * factor
                    count += 1

        if count > 0:
            sep_x /= count
            sep_y /= count

        return sep_x, sep_y

    def apply_swarm_rules(self, agents: List['Agent']):
        # Different radii for different behaviors
        separation_radius = 30 if self.role == 'searching' else 20
        cohesion_radius = 50 if self.carrying_resource else 80
        alignment_radius = 40

        separate_x = separate_y = 0
        cohesion_x = cohesion_y = 0
        align_x = align_y = 0

        relevant_agents = 0

        for other in agents:
            if other != self and other.team == self.team:
                distance = self.distance_to(other.x, other.y)

                # Only apply cohesion with agents in same role
                same_role = (other.role == self.role)

                # Enhanced separation for searching agents
                if distance < separation_radius:
                    factor = 1.0 - (distance / separation_radius)
                    if self.role == 'searching':
                        factor *= 2.0  # Stronger separation when searching
                    separate_x += (self.x - other.x) * factor
                    separate_y += (self.y - other.y) * factor

                # Cohesion only with relevant agents
                if distance < cohesion_radius and same_role:
                    # Reduce cohesion if we're searching and there are too many nearby
                    if self.role == 'searching':
                        nearby_count = sum(1 for a in agents
                                           if a.team == self.team and
                                           self.distance_to(a.x, a.y) < cohesion_radius)
                        if nearby_count > 3:
                            continue

                    cohesion_x += other.x
                    cohesion_y += other.y
                    relevant_agents += 1

                # Alignment mainly for agents in same role
                if distance < alignment_radius and same_role:
                    align_x += other.velocity_x
                    align_y += other.velocity_y

        # Calculate final velocities
        if relevant_agents > 0:
            cohesion_x /= relevant_agents
            cohesion_y /= relevant_agents
            align_x /= relevant_agents
            align_y /= relevant_agents

            # Adjust weights based on role and situation
            sep_weight = 0.5 if self.role == 'searching' else 0.3
            coh_weight = 0.1 if self.role == 'searching' else 0.3
            ali_weight = 0.2

            # Reduce cohesion if there are too many nearby agents
            if self.should_separate_from_group(agents):
                coh_weight *= 0.2
                sep_weight *= 2.0

            self.velocity_x += (separate_x * sep_weight +
                                (cohesion_x - self.x) * coh_weight +
                                align_x * ali_weight)
            self.velocity_y += (separate_y * sep_weight +
                                (cohesion_y - self.y) * coh_weight +
                                align_y * ali_weight)

            # Normalize velocity
            speed = math.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
            if speed > self.speed:
                self.velocity_x = (self.velocity_x / speed) * self.speed
                self.velocity_y = (self.velocity_y / speed) * self.speed

    def update(self, resources: List[Resource], agents: List['Agent'], base_x: float, base_y: float):
        self.nearby_agents = agents  # Store for use in other methods

        # Update position based on current velocity
        self.x += self.velocity_x
        self.y += self.velocity_y

        # Keep within bounds
        self.x = max(0, min(self.x, WINDOW_WIDTH))
        self.y = max(0, min(self.y, WINDOW_HEIGHT))

        # Check if agent is stuck
        current_pos = (self.x, self.y)
        distance_moved = math.sqrt(
            (current_pos[0] - self.last_position[0]) ** 2 +
            (current_pos[1] - self.last_position[1]) ** 2
        )

        if distance_moved < 0.5:  # If barely moving
            self.stuck_time += 1
            if self.stuck_time > 60:  # If stuck for 1 second
                # Add random movement to break free
                self.velocity_x += random.uniform(-1, 1)
                self.velocity_y += random.uniform(-1, 1)
                if self.role == 'searching':
                    self.target = None  # Reset target if searching
        else:
            self.stuck_time = 0

        self.last_position = current_pos

        # Decrease energy over time
        self.energy = max(0, self.energy - 0.01)

        # Handle team-specific behavior
        if self.team == 'collector':
            if self.carrying_resource:
                # Return to base with resource
                self.role = 'returning'
                self.move_towards(base_x, base_y)

                # Update resource position if part of carrier group
                if len(self.carrying_resource.carriers) > 0:
                    avg_x = sum(agent.x for agent in self.carrying_resource.carriers) / len(
                        self.carrying_resource.carriers)
                    avg_y = sum(agent.y for agent in self.carrying_resource.carriers) / len(
                        self.carrying_resource.carriers)
                    self.carrying_resource.x = avg_x
                    self.carrying_resource.y = avg_y
            else:
                if self.role == 'searching':
                    # Only evaluate resources if not too many agents nearby
                    nearby_searching = sum(1 for agent in agents
                                           if agent.team == self.team and
                                           agent.role == 'searching' and
                                           self.distance_to(agent.x, agent.y) < 50)

                    if nearby_searching <= 3:  # Only search if not too crowded
                        best_value = 0
                        best_resource = None

                        for resource in resources:
                            value = self.evaluate_resource(resource, agents)
                            if value > best_value:
                                best_value = value
                                best_resource = resource

                        if best_resource:
                            self.target = best_resource
                            self.role = 'gathering'
                    else:
                        # Move away from the group if too crowded
                        sep_x, sep_y = self.get_separation_vector(agents)
                        if sep_x != 0 or sep_y != 0:
                            magnitude = math.sqrt(sep_x ** 2 + sep_y ** 2)
                            self.velocity_x = (sep_x / magnitude) * self.speed
                            self.velocity_y = (sep_y / magnitude) * self.speed

                if self.role == 'gathering' and self.target:
                    if not self.target.collected:
                        self.move_towards(self.target.x, self.target.y)

                        if self.distance_to(self.target.x, self.target.y) < 20:
                            self.waiting_time += 1
                            if self.waiting_time > 60:
                                self.role = 'searching'
                                self.target = None
                                self.waiting_time = 0
                    else:
                        self.role = 'searching'
                        self.target = None

        elif self.team == 'opposer':
            target, score = self.find_best_target(agents)

            if target:
                self.update_opposer_movement(target)
            else:
                # Search pattern when no targets visible
                if not hasattr(self, 'patrol_point') or self.distance_to(self.patrol_point[0],
                                                                         self.patrol_point[1]) < 20:
                    # Generate new patrol point, biased towards the center and away from edges
                    center_weight = 0.3
                    self.patrol_point = (
                        random.uniform(WINDOW_WIDTH * 0.1, WINDOW_WIDTH * 0.9) * (1 - center_weight) +
                        (WINDOW_WIDTH / 2) * center_weight,
                        random.uniform(WINDOW_HEIGHT * 0.1, WINDOW_HEIGHT * 0.8) * (1 - center_weight) +
                        (WINDOW_HEIGHT / 2) * center_weight
                    )
                self.move_towards(*self.patrol_point)
                self.speed = 1.5  # Slower patrol speed to conserve energy

        # Apply swarm behavior rules
        self.apply_swarm_rules(agents)

    def find_best_target(self, agents: list[Agent]) -> tuple[Agent, float]:
        """Find the most valuable target collector based on multiple factors."""
        best_target = None
        best_score = float('-inf')

        for agent in agents:
            if agent.team == 'collector' and agent.carrying_resource:
                score = self.evaluate_target(agent)
                if score > best_score:
                    best_score = score
                    best_target = agent

        return best_target, best_score

    def evaluate_target(self, collector: 'Agent') -> float:
        """Evaluate a collector as a potential target based on multiple factors."""
        if not collector.carrying_resource:
            return float('-inf')

        distance = self.distance_to(collector.x, collector.y)
        if distance > self.perception_radius:
            return float('-inf')

        # Base score is resource value
        score = collector.carrying_resource.value

        # Adjust for distance (closer is better)
        score *= (1.0 - (distance / self.perception_radius))

        # Consider the number of carriers (more carriers = more valuable target)
        carrier_count = len(collector.carrying_resource.carriers)
        score *= (1 + (carrier_count * 0.2))

        # Consider distance to base (further from base is better opportunity)
        distance_to_base = collector.distance_to(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30)
        score *= (1 + (distance_to_base / WINDOW_WIDTH) * 0.5)

        # Consider nearby opposers for coordination
        nearby_opposers = self.count_nearby_opposers()
        if nearby_opposers > 0:
            score *= (1 + (nearby_opposers * 0.15))

        return score

    def count_nearby_opposers(self) -> int:
        """Count number of nearby opposer agents for coordination."""
        count = 0
        for agent in self.nearby_agents:
            if (agent.team == 'opposer' and
                    agent != self and
                    self.distance_to(agent.x, agent.y) < self.communication_radius):
                count += 1
        return count

    def update_opposer_movement(self, target: 'Agent'):
        """Update opposer movement strategy based on target position and team coordination."""
        if not target:
            return

        # Calculate interception point based on target's velocity
        prediction_steps = 10
        intercept_x = target.x + (target.velocity_x * prediction_steps)
        intercept_y = target.y + (target.velocity_y * prediction_steps)

        # Adjust for team coordination
        nearby_opposers = []
        for agent in self.nearby_agents:
            if (agent.team == 'opposer' and
                    agent != self and
                    self.distance_to(agent.x, agent.y) < self.communication_radius):
                nearby_opposers.append(agent)

        if nearby_opposers:
            # Spread out opposers for surrounding the target
            angle_offset = (2 * math.pi) / (len(nearby_opposers) + 1)
            my_index = nearby_opposers.index(self) if self in nearby_opposers else 0

            # Calculate surrounding position
            radius = 30
            surround_x = intercept_x + math.cos(angle_offset * my_index) * radius
            surround_y = intercept_y + math.sin(angle_offset * my_index) * radius

            self.move_towards(surround_x, surround_y)
        else:
            # Direct pursuit if no nearby opposers
            self.move_towards(intercept_x, intercept_y)

        # Adjust speed based on distance and target value
        distance = self.distance_to(target.x, target.y)
        if distance < 100:
            self.speed = min(3.0, 2.0 + (target.carrying_resource.value * 0.1))
        else:
            self.speed = 2.0

    def draw(self, screen):
        color = BLUE if self.team == 'collector' else RED
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.size)

        # Draw energy bar
        energy_bar_length = 20
        energy_bar_height = 3
        energy_percentage = self.energy / 100
        pygame.draw.rect(screen, RED, (
            int(self.x - energy_bar_length / 2),
            int(self.y - self.size - 5),
            energy_bar_length,
            energy_bar_height
        ))
        pygame.draw.rect(screen, GREEN, (
            int(self.x - energy_bar_length / 2),
            int(self.y - self.size - 5),
            int(energy_bar_length * energy_percentage),
            energy_bar_height
        ))
class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Enhanced Swarm Resource Collection Simulation")
        self.clock = pygame.time.Clock()

        # Initialize agents
        self.collectors = [Agent(random.randint(0, WINDOW_WIDTH),
                                 random.randint(0, WINDOW_HEIGHT),
                                 'collector') for _ in range(15)]
        self.opposers = [Agent(random.randint(0, WINDOW_WIDTH),
                               random.randint(0, WINDOW_HEIGHT),
                               'opposer') for _ in range(5)]
        self.agents = self.collectors + self.opposers

        # Initialize resources with different types
        self.resources = []
        for _ in range(10):
            resource_type = random.choice(list(ResourceType))
            self.resources.append(Resource(
                random.randint(0, WINDOW_WIDTH),
                random.randint(0, WINDOW_HEIGHT),
                resource_type
            ))

        self.base_x = WINDOW_WIDTH // 2
        self.base_y = WINDOW_HEIGHT - 30

        self.collected_resources = 0
        self.stolen_resources = 0
        self.total_value = 0

    def handle_collisions(self):
        # Track resources to remove after collection
        resources_to_remove = []

        # Check collector-resource collisions
        for collector in self.collectors:
            if not collector.carrying_resource:
                for resource in self.resources:
                    if (not resource.collected and
                            collector.distance_to(resource.x, resource.y) < 15):

                        # Check if we have enough agents nearby
                        nearby_collectors = [
                            agent for agent in self.collectors
                            if (agent != collector and
                                not agent.carrying_resource and
                                agent.distance_to(resource.x, resource.y) < 20)
                        ]

                        if len(nearby_collectors) + 1 >= resource.type.value:
                            # Enough agents to carry the resource
                            resource.collected = True
                            resource.carriers = [collector] + nearby_collectors[:resource.type.value - 1]
                            for carrier in resource.carriers:
                                carrier.carrying_resource = resource
                                carrier.role = 'returning'  # Set role to returning
                        break

        # Check opposer-collector collisions
        for opposer in self.opposers:
            for collector in self.collectors:
                if (collector.carrying_resource and
                        opposer.distance_to(collector.x, collector.y) < 15):
                    resource = collector.carrying_resource
                    # Release all carriers
                    for carrier in resource.carriers:
                        carrier.carrying_resource = None
                        carrier.role = 'searching'  # Reset role to searching
                    resource.collected = False
                    resource.carriers = []
                    self.stolen_resources += 0.01

        # Check collector-base collisions and process resource collection
        for resource in self.resources:
            if resource.collected and resource.carriers:
                # Check if ALL carriers are near the base
                all_carriers_at_base = all(
                    carrier.distance_to(self.base_x, self.base_y) < 20
                    for carrier in resource.carriers
                )

                if all_carriers_at_base:
                    # Add to score and mark resource for removal
                    self.collected_resources += 1
                    self.total_value += resource.value
                    resources_to_remove.append(resource)

                    # Release all carriers
                    for carrier in resource.carriers:
                        carrier.carrying_resource = None
                        carrier.role = 'searching'  # Reset role to searching
                        # Add some randomization to prevent base camping
                        angle = random.uniform(0, 2 * math.pi)
                        distance = random.uniform(50, 100)
                        carrier.x = self.base_x + math.cos(angle) * distance
                        carrier.y = self.base_y + math.sin(angle) * distance

        # Remove collected resources
        for resource in resources_to_remove:
            if resource in self.resources:
                self.resources.remove(resource)

    def run(self):
        running = True
        spawn_timer = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update spawn timer
            spawn_timer += 1

            # Spawn new resources periodically (every 3 seconds)
            if spawn_timer >= 180:  # 180 frames at 60 FPS = 3 seconds
                resource_type = random.choice(list(ResourceType))
                # Spawn resources away from the base
                while True:
                    x = random.randint(0, WINDOW_WIDTH)
                    y = random.randint(0, WINDOW_HEIGHT - 100)  # Keep away from base
                    # Check distance from base
                    if math.sqrt((x - self.base_x) ** 2 + (y - self.base_y) ** 2) > 150:
                        break

                self.resources.append(Resource(x, y, resource_type))
                spawn_timer = 0

            # Update agents and handle collisions
            for agent in self.agents:
                agent.update(self.resources, self.agents, self.base_x, self.base_y)

            self.handle_collisions()


            # Spawn new resources occasionally
            if random.random() < 0.002:  # 0.2% chance each frame
                resource_type = random.choice(list(ResourceType))
                self.resources.append(Resource(
                    random.randint(0, WINDOW_WIDTH),
                    random.randint(0, WINDOW_HEIGHT),
                    resource_type
                ))

            # Draw
            self.screen.fill(BLACK)

            # Draw base
            pygame.draw.rect(self.screen, WHITE,
                             (self.base_x - 20, self.base_y - 10, 40, 20))

            # Draw resources
            for resource in self.resources:
                resource.draw(self.screen)

            # Draw all agents
            for agent in self.agents:
                agent.draw(self.screen)

                # Draw lines between collaborating agents
                if agent.carrying_resource and agent.team == 'collector':
                    for collaborator in agent.carrying_resource.carriers:
                        if collaborator != agent:
                            pygame.draw.line(self.screen, WHITE,
                                             (int(agent.x), int(agent.y)),
                                             (int(collaborator.x), int(collaborator.y)), 1)

            # Draw UI and statistics
            font = pygame.font.Font(None, 36)

            # Draw resource type legend
            pygame.draw.circle(self.screen, GREEN, (30, WINDOW_HEIGHT - 60), 5)
            pygame.draw.circle(self.screen, YELLOW, (30, WINDOW_HEIGHT - 40), 5)
            pygame.draw.circle(self.screen, PURPLE, (30, WINDOW_HEIGHT - 20), 5)

            legend_font = pygame.font.Font(None, 24)
            light_text = legend_font.render("Light (1 agent)", True, WHITE)
            medium_text = legend_font.render("Medium (2 agents)", True, WHITE)
            heavy_text = legend_font.render("Heavy (3 agents)", True, WHITE)

            self.screen.blit(light_text, (45, WINDOW_HEIGHT - 70))
            self.screen.blit(medium_text, (45, WINDOW_HEIGHT - 50))
            self.screen.blit(heavy_text, (45, WINDOW_HEIGHT - 30))

            # Draw statistics
            collected_text = font.render(f'Resources Collected: {self.collected_resources}', True, WHITE)
            stolen_text = font.render(f'Resources Stolen: {self.stolen_resources}', True, WHITE)
            value_text = font.render(f'Total Value: {self.total_value}', True, WHITE)

            # Count active resources by type
            active_resources = {
                ResourceType.LIGHT: 0,
                ResourceType.MEDIUM: 0,
                ResourceType.HEAVY: 0
            }
            for resource in self.resources:
                if not resource.collected:
                    active_resources[resource.type] += 1

            resources_text = font.render(
                f'Available - L:{active_resources[ResourceType.LIGHT]} ' +
                f'M:{active_resources[ResourceType.MEDIUM]} ' +
                f'H:{active_resources[ResourceType.HEAVY]}', True, WHITE)

            self.screen.blit(collected_text, (10, 10))
            self.screen.blit(stolen_text, (10, 50))
            self.screen.blit(value_text, (10, 90))
            self.screen.blit(resources_text, (10, 130))

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()


# tend to stuck after some time