import math
import json
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pygame

CONFIG = {
    "window_w": 1000,
    "window_h": 800,
    "fps": 60,

    "arena_radius": 320,
    "arena_margin": 60,

    "fish_count": 60,
    "predator_count": 1,

    "sim_seconds": 30.0,

    # Speeds are in pixels per second
    "fish_speed": 120.0,
    "predator_speed": 165.0,   # slightly faster than fish
    "predator_turn_rate": 9.0, # radians/sec, limits how sharply predator can turn

    "fish_turn_rate": 6.0,     # radians/sec, how fast fish can turn

    "eat_radius": 16.0,        # predator eats fish if within this distance

    # Evolution
    "generations": 200,
    "elitism_frac": 0.20,      # top % kept
    "mutation_sigma": 0.10,    # noise added to weights
    "mutation_rate": 1.00,     # probability each child gets mutated (keep 1.0 initially)

    # Obstacles
    "obstacle_count": 3,
    "obstacle_radius": 20.0,

    # Network architecture
    "input_dim": 10,
    "hidden_dim": 12,
    "output_dim": 2,           # turn and speed (both are outputs)

    # Fitness weights
    "wall_hit_penalty": 0.25,
    "jerk_penalty": 0.001,     
    "alive_bonus": 1.0,        
    "speed_efficiency": 0.01,  
    "speed_penalty": 0.005,    
}

# Utility math

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def vec_len(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)

def normalize(x: float, y: float) -> Tuple[float, float]:
    n = vec_len(x, y)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, y / n

def angle_of(x: float, y: float) -> float:
    return math.atan2(y, x)

def wrap_angle(a: float) -> float:
    # wrap to [-pi, pi]
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# Simple Neural Net 
class TinyNet:
    """
    Feedforward net: input -> hidden(tanh) -> output(tanh)
    We store parameters as a single flat vector "genome".
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, genome: Optional[np.ndarray] = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Parameter counts:
        # W1: hidden x input
        # b1: hidden
        # W2: output x hidden
        # b2: output
        self.n_params = hidden_dim * input_dim + hidden_dim + output_dim * hidden_dim + output_dim

        if genome is None:
            self.genome = np.random.randn(self.n_params).astype(np.float32) * 0.5
        else:
            assert genome.shape == (self.n_params,)
            self.genome = genome.astype(np.float32)

        self._unpack_views()

    def _unpack_views(self) -> None:
        i, h, o = self.input_dim, self.hidden_dim, self.output_dim
        idx = 0
        w1_n = h * i
        self.W1 = self.genome[idx:idx + w1_n].reshape(h, i); idx += w1_n
        self.b1 = self.genome[idx:idx + h]; idx += h
        w2_n = o * h
        self.W2 = self.genome[idx:idx + w2_n].reshape(o, h); idx += w2_n
        self.b2 = self.genome[idx:idx + o]; idx += o

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x shape: (input_dim,)
        z1 = np.tanh(self.W1 @ x + self.b1)          # (hidden_dim,)
        z2 = np.tanh(self.W2 @ z1 + self.b2)         # (output_dim,)
        return z2

    def copy(self) -> "TinyNet":
        return TinyNet(self.input_dim, self.hidden_dim, self.output_dim, genome=self.genome.copy())

# Entities

@dataclass
class Fish:
    x: float
    y: float
    heading: float
    alive: bool = True

    # Stats for fitness
    survival_time: float = 0.0
    wall_hits: int = 0
    jerk_accum: float = 0.0
    last_turn: float = 0.0
    pred_dist_accum: float = 0.0
    turn_effort: float = 0.0
    
    speed_when_close: float = 0.0  
    speed_when_far: float = 0.0    
    close_time: float = 0.0
    far_time: float = 0.0

@dataclass
class Predator:
    x: float
    y: float
    heading: float

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float

# Simulation

class World:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.cx = cfg["window_w"] // 2
        self.cy = cfg["window_h"] // 2
        self.R = cfg["arena_radius"]

        self.fish_speed = cfg["fish_speed"]
        self.pred_speed = cfg["predator_speed"]
        self.fish_turn_rate = cfg["fish_turn_rate"]
        self.pred_turn_rate = cfg["predator_turn_rate"]
        self.eat_radius = cfg["eat_radius"]

        self.fish: List[Fish] = []
        self.preds: List[Predator] = []

        self.obstacles: List[Obstacle] = []
        self.obstacle_radius = cfg["obstacle_radius"]

    def reset(self, fish_nets: List[TinyNet], seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.fish.clear()
        self.preds.clear()

        # Spawn obstacles
        self.obstacles.clear()
        for _ in range(self.cfg["obstacle_count"]):
            x, y = self._rand_point_in_circle(self.cx, self.cy, self.R * 0.8)
            self.obstacles.append(Obstacle(x=x, y=y, radius=self.obstacle_radius))

        # Spawn fish randomly inside circle
        for _ in fish_nets:
            x, y = self._rand_point_in_circle(self.cx, self.cy, self.R * 0.9)
            heading = random.uniform(-math.pi, math.pi)
            self.fish.append(Fish(x=x, y=y, heading=heading, alive=True))

        # Spawn predators closer to center
        for _ in range(self.cfg["predator_count"]):
            x, y = self._rand_point_in_circle(self.cx, self.cy, self.R * 0.3)
            heading = random.uniform(-math.pi, math.pi)
            self.preds.append(Predator(x=x, y=y, heading=heading))

    def _push_inside_obstacles(self, entity_xy: Tuple[float, float]) -> Tuple[float, float, bool]:
        x, y = entity_xy
        pushed = False
        for obs in self.obstacles:
            dx = x - obs.x
            dy = y - obs.y
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = obs.radius + 1.0  # small buffer
            if dist < min_dist:
                # Push out
                if dist > 0:
                    nx, ny = dx / dist, dy / dist
                    x = obs.x + nx * obs.radius
                    y = obs.y + ny * obs.radius
                return x, y, True
        return x, y, False

    def _rand_point_in_circle(self, cx: float, cy: float, radius: float) -> Tuple[float, float]:
        # Rejection sampling
        for _ in range(1000):
            rx = random.uniform(-radius, radius)
            ry = random.uniform(-radius, radius)
            if rx * rx + ry * ry <= radius * radius:
                return cx + rx, cy + ry
        return cx, cy

    def _inside_arena(self, x: float, y: float) -> bool:
        dx = x - self.cx
        dy = y - self.cy
        return dx * dx + dy * dy <= self.R * self.R

    def _push_inside_arena(self, entity_xy: Tuple[float, float]) -> Tuple[float, float, bool]:
        x, y = entity_xy
        dx = x - self.cx
        dy = y - self.cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= self.R:
            return x, y, False
        # Clamp back onto circle
        nx, ny = dx / dist, dy / dist
        x = self.cx + nx * (self.R - 1.0)
        y = self.cy + ny * (self.R - 1.0)
        return x, y, True

    def step(self, fish_nets: List[TinyNet], dt: float) -> None:
        # Predator logic first (chase nearest alive fish)
        alive_indices = [i for i, f in enumerate(self.fish) if f.alive]

        for p in self.preds:
            if not alive_indices:
                # drift
                p.heading += random.uniform(-0.2, 0.2) * dt
            else:
                # nearest alive fish
                best_i = None
                best_d2 = 1e18
                for i in alive_indices:
                    f = self.fish[i]
                    dx = f.x - p.x
                    dy = f.y - p.y
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best_i = i

                target = self.fish[best_i]
                desired = angle_of(target.x - p.x, target.y - p.y)
                # turn limited
                delta = wrap_angle(desired - p.heading)
                max_turn = self.pred_turn_rate * dt
                delta = clamp(delta, -max_turn, max_turn)
                p.heading = wrap_angle(p.heading + delta)

            # Move predator
            p.x += math.cos(p.heading) * self.pred_speed * dt
            p.y += math.sin(p.heading) * self.pred_speed * dt

            p.x, p.y, _ = self._push_inside_arena((p.x, p.y))

        # Fish logic (evolved nets)
        for idx, (f, net) in enumerate(zip(self.fish, fish_nets)):
            if not f.alive:
                continue

            # Sensors
            inp = self._fish_inputs(f)

            # Policy output: turn in [-1,1] scaled by fish_turn_rate
            out = net.forward(inp)

            turn_cmd = float(out[0])  # -1 to 1
            speed_cmd = float(out[1])  # -1 to 1

            turn = turn_cmd * self.fish_turn_rate
            f.turn_effort += abs(turn) * dt

            applied_turn = turn * dt
            f.heading = wrap_angle(f.heading + applied_turn)

            # Jerk (how rapidly turn changed)
            f.jerk_accum += abs(turn - f.last_turn)
            f.last_turn = turn

            speed_mul = 0.5 + (speed_cmd + 1.0) * 0.4  # map [-1,1] to [0.5,1.3]
            speed = self.fish_speed * speed_mul

            best_d = float("inf")
            for p in self.preds:
                d = math.hypot(p.x - f.x, p.y - f.y)
                if d < best_d:
                    best_d = d
            
            # Define "close" as within 150 pixels
            danger_threshold = 150.0
            if best_d < danger_threshold:
                f.speed_when_close += speed_mul * dt
                f.close_time += dt
            else:
                f.speed_when_far += speed_mul * dt
                f.far_time += dt
            
            f.pred_dist_accum += best_d * dt

            # Move
            f.x += math.cos(f.heading) * speed * dt
            f.y += math.sin(f.heading) * speed * dt

            # Keep inside arena; count wall hits if clamped
            f.x, f.y, hit = self._push_inside_arena((f.x, f.y))
            if hit:
                f.wall_hits += 1

            f.x, f.y, obs_hit = self._push_inside_obstacles((f.x, f.y))

            # Survival time
            f.survival_time += dt

        # Eating check
        for p in self.preds:
            for f in self.fish:
                if not f.alive:
                    continue
                if vec_len(f.x - p.x, f.y - p.y) <= self.eat_radius:
                    f.alive = False

    def _fish_inputs(self, f: Fish) -> np.ndarray:
        """
        Returns normalized input vector length = CONFIG['input_dim'].
        Inputs (8):
          0: vx_norm
          1: vy_norm
          2: dist_pred_norm (closest predator)
          3: sin(angle_to_pred)
          4: cos(angle_to_pred)
          5: dist_wall_norm
          6: sin(heading)
          7: cos(heading)
        """
        # self velocity direction
        vx = math.cos(f.heading)
        vy = math.sin(f.heading)

        # closest predator
        best_d = 1e18
        best_ang = 0.0
        for p in self.preds:
            dx = p.x - f.x
            dy = p.y - f.y
            d = vec_len(dx, dy)
            if d < best_d:
                best_d = d
                best_ang = angle_of(dx, dy)

        # normalize distance to predator: 0 near predator, 1 far away
        dist_pred = clamp(best_d / (self.R * 2.0), 0.0, 1.0)

        # relative angle to predator
        rel = wrap_angle(best_ang - f.heading)
        s_rel = math.sin(rel)
        c_rel = math.cos(rel)

        # distance to wall (0 at wall, 1 at center-ish)
        dx0 = f.x - self.cx
        dy0 = f.y - self.cy
        dist_center = vec_len(dx0, dy0)
        dist_wall = clamp((self.R - dist_center) / self.R, 0.0, 1.0)

        best_obs_d = 1e18
        best_obs_ang = 0.0
        for obs in self.obstacles:
            dx = obs.x - f.x
            dy = obs.y - f.y
            d = vec_len(dx, dy)
            if d < best_obs_d:
                best_obs_d = d
                best_obs_ang = angle_of(dx, dy)

        dist_obs = clamp(best_obs_d / (self.R * 0.5), 0.0, 1.0)
        rel_obs = wrap_angle(best_obs_ang - f.heading)
        s_obs = math.sin(rel_obs)
        c_obs = math.cos(rel_obs)

        inp = np.array([
            vx, vy,
            dist_pred,
            s_rel, c_rel,
            dist_wall,
            math.sin(f.heading),
            math.cos(f.heading),
            dist_obs,
            s_obs, 
            c_obs,
        ], dtype=np.float32)

        return inp

# Evolution

def compute_fitness(f: Fish, cfg: Dict) -> float:
    fitness = f.survival_time * cfg["alive_bonus"]
    fitness -= cfg["wall_hit_penalty"] * f.wall_hits
    fitness -= cfg["jerk_penalty"] * f.jerk_accum
    fitness += 0.002 * f.pred_dist_accum
    fitness -= 0.05 * f.turn_effort
    
    # Reward high speed when close to predator
    if f.close_time > 0:
        avg_speed_close = f.speed_when_close / f.close_time
        fitness += cfg["speed_efficiency"] * avg_speed_close * f.close_time
    
    # Penalize constant high speed when far from predator (energy conservation)
    if f.far_time > 0:
        avg_speed_far = f.speed_when_far / f.far_time
        fitness -= cfg["speed_penalty"] * avg_speed_far * f.far_time
    
    return float(fitness)

def mutate(genome: np.ndarray, sigma: float) -> np.ndarray:
    child = genome.copy()
    noise = np.random.randn(child.shape[0]).astype(np.float32) * sigma
    child += noise
    return child

def save_best(path: str, cfg: Dict, net: TinyNet, gen: int, best_fit: float) -> None:
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": gen,
        "best_fitness": best_fit,
        "config": cfg,
        "net": {
            "input_dim": net.input_dim,
            "hidden_dim": net.hidden_dim,
            "output_dim": net.output_dim,
            "genome": net.genome.tolist(),
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_best(path: str) -> Tuple[Dict, TinyNet]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    net_info = payload["net"]
    net = TinyNet(
        net_info["input_dim"],
        net_info["hidden_dim"],
        net_info["output_dim"],
        genome=np.array(net_info["genome"], dtype=np.float32),
    )
    return payload, net

# Rendering
def draw_world(screen: pygame.Surface, world: World, gen: int, t_left: float, best_fit: float, avg_fit: float, mode: str) -> None:
    screen.fill((16, 18, 24))

    # Arena
    pygame.draw.circle(screen, (60, 70, 90), (world.cx, world.cy), world.R, width=3)
    
    # Draw danger zone (150px radius around predators)
    for p in world.preds:
        pygame.draw.circle(screen, (80, 40, 40), (int(p.x), int(p.y)), 150, width=1)

    for obs in world.obstacles:
        pygame.draw.circle(screen, (100, 100, 100), (int(obs.x), int(obs.y)), int(obs.radius))
    # Fish - color by speed
    for f in world.fish:
        if not f.alive:
            continue
        
        # Calculate approximate speed for coloring
        # Fish that move more get brighter
        brightness = 180 + int(20 * (f.survival_time / 10))  # subtle brightness increase
        pygame.draw.circle(screen, (brightness, 220, 255), (int(f.x), int(f.y)), 4)

        # tiny heading line
        hx = f.x + math.cos(f.heading) * 10
        hy = f.y + math.sin(f.heading) * 10
        pygame.draw.line(screen, (120, 160, 200), (int(f.x), int(f.y)), (int(hx), int(hy)), width=2)

    # Predator
    for p in world.preds:
        pygame.draw.circle(screen, (255, 90, 90), (int(p.x), int(p.y)), 10)
        hx = p.x + math.cos(p.heading) * 18
        hy = p.y + math.sin(p.heading) * 18
        pygame.draw.line(screen, (255, 140, 140), (int(p.x), int(p.y)), (int(hx), int(hy)), width=3)

    # HUD
    font = pygame.font.SysFont("consolas", 18)
    alive = sum(1 for f in world.fish if f.alive)

    lines = [
        f"Mode: {mode}   (SPACE: toggle train/replay)   (R: reset replay)   (ESC: quit)",
        f"Gen: {gen}   Alive: {alive}/{len(world.fish)}   Time left: {t_left:5.1f}s",
        f"BestFit: {best_fit:8.2f}   AvgFit: {avg_fit:8.2f}",
        f"Red circle = danger zone. Fish should speed up inside it!",
    ]
    y = 12
    for ln in lines:
        surf = font.render(ln, True, (230, 235, 245))
        screen.blit(surf, (12, y))
        y += 22

# Main loop
def run_training(cfg: Dict) -> None:
    pygame.init()
    screen = pygame.display.set_mode((cfg["window_w"], cfg["window_h"]))
    pygame.display.set_caption("Evolving Fish Survival (Neuroevolution)")
    clock = pygame.time.Clock()

    input_dim = cfg["input_dim"]
    hidden_dim = cfg["hidden_dim"]
    output_dim = cfg["output_dim"]

    # Initialize population
    population: List[TinyNet] = [TinyNet(input_dim, hidden_dim, output_dim) for _ in range(cfg["fish_count"])]

    world = World(cfg)

    gen = 1
    best_overall_fit = -1e18
    best_overall_net = population[0].copy()

    # Train/replay toggle
    mode = "TRAIN"  # or "REPLAY"
    replay_net = best_overall_net.copy()

    def start_episode(nets: List[TinyNet], seed: Optional[int] = None):
        world.reset(nets, seed=seed)

    start_episode(population)

    sim_time = cfg["sim_seconds"]
    t = 0.0

    # Store last gen fitness for display
    last_best = 0.0
    last_avg = 0.0

    while True:
        dt = clock.tick(cfg["fps"]) / 1000.0

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_SPACE:
                    if mode == "TRAIN":
                        mode = "REPLAY"
                        replay_net = best_overall_net.copy()
                        start_episode([replay_net] * cfg["fish_count"], seed=42)
                        t = 0.0
                    else:
                        mode = "TRAIN"
                        start_episode(population)
                        t = 0.0
                if event.key == pygame.K_r and mode == "REPLAY":
                    start_episode([replay_net] * cfg["fish_count"], seed=42)
                    t = 0.0

        # Step sim
        if mode == "TRAIN":
            world.step(population, dt)
        else:
            # In replay, all fish share the same best policy so you can see behavior clearly
            world.step([replay_net] * cfg["fish_count"], dt)

        t += dt
        t_left = max(0.0, sim_time - t)

        # End episode?
        if t >= sim_time or sum(1 for f in world.fish if f.alive) == 0:
            # Compute fitness
            fits = [compute_fitness(f, cfg) for f in world.fish]
            last_best = float(np.max(fits)) if fits else 0.0
            last_avg = float(np.mean(fits)) if fits else 0.0

            if mode == "TRAIN":
                # Track best net
                best_idx = int(np.argmax(fits))
                best_net = population[best_idx].copy()

                if last_best > best_overall_fit:
                    best_overall_fit = last_best
                    best_overall_net = best_net.copy()
                    save_best("runs/best.json", cfg, best_overall_net, gen, best_overall_fit)
                    print(f"[Gen {gen:4d}] NEW BEST: {best_overall_fit:.2f}  (saved runs/best.json)")

                # Selection
                elite_n = max(2, int(cfg["elitism_frac"] * len(population)))
                order = np.argsort(fits)[::-1]  # descending
                elites = [population[i].copy() for i in order[:elite_n]]

                # Reproduce
                new_pop: List[TinyNet] = []
                # Keep elites
                new_pop.extend([e.copy() for e in elites])

                # Fill rest with mutated children of elites
                while len(new_pop) < len(population):
                    parent = random.choice(elites)
                    child = parent.copy()
                    if random.random() < cfg["mutation_rate"]:
                        child.genome = mutate(child.genome, cfg["mutation_sigma"])
                        child._unpack_views()
                    new_pop.append(child)

                population = new_pop
                gen += 1

                # Stop if done
                if gen > cfg["generations"]:
                    print("Training complete. Best saved at runs/best.json")
                    # Auto-switch to replay at end
                    mode = "REPLAY"
                    replay_net = best_overall_net.copy()
                    start_episode([replay_net] * cfg["fish_count"], seed=42)
                    t = 0.0
                else:
                    # Next episode
                    start_episode(population)
                    t = 0.0
            else:
                # Replay: just reset episode for looping
                start_episode([replay_net] * cfg["fish_count"], seed=42)
                t = 0.0

        # Render
        draw_world(screen, world, gen if mode == "TRAIN" else -1, t_left, best_overall_fit, last_avg, mode)
        pygame.display.flip()

def run_replay(best_path: str = "runs/best.json") -> None:
    payload, net = load_best(best_path)
    cfg = payload.get("config", CONFIG)
    # force fish_count from config if missing
    cfg["fish_count"] = cfg.get("fish_count", CONFIG["fish_count"])

    pygame.init()
    screen = pygame.display.set_mode((cfg["window_w"], cfg["window_h"]))
    pygame.display.set_caption("Evolving Fish Survival (Replay)")
    clock = pygame.time.Clock()

    world = World(cfg)

    def start_episode():
        world.reset([net] * cfg["fish_count"], seed=42)

    start_episode()
    t = 0.0
    sim_time = cfg["sim_seconds"]

    while True:
        dt = clock.tick(cfg["fps"]) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_r:
                    start_episode()
                    t = 0.0

        world.step([net] * cfg["fish_count"], dt)
        t += dt

        if t >= sim_time or sum(1 for f in world.fish if f.alive) == 0:
            start_episode()
            t = 0.0

        draw_world(screen, world, payload.get("generation", -1), max(0.0, sim_time - t),
                   payload.get("best_fitness", 0.0), 0.0, "REPLAY")
        pygame.display.flip()

if __name__ == "__main__":
    # Default: train
    run_training(CONFIG)
    # If you only want replay, comment the line above and uncomment:
    # run_replay("runs/best.json")