"""
AI Snake - Project 441
TOBB ETÜ - Mehmet Şamil ARAT - 211401002

Structure:
  - SnakeGame      : core game engine (no UI dependency)
  - BFSAgent       : baseline agent (BFS + safety check + tail-chase)
  - QLearningAgent : improved agent (tabular Q-learning, 2048-state table)
  - run_game()     : pygame visualizer
  - train()        : training loop for Q-agent
  - evaluate()     : headless evaluation over N seeds
  - compare()      : run both agents, side-by-side stats

Usage:
  python snake_ai.py                         # watch BFS agent
  python snake_ai.py --mode ai --agent ql    # watch trained Q-agent
  python snake_ai.py --mode train            # train Q-agent (~5000 eps)
  python snake_ai.py --mode eval             # evaluate BFS (100 seeds)
  python snake_ai.py --mode eval --agent ql  # evaluate Q-agent
  python snake_ai.py --mode compare          # side-by-side comparison
  python snake_ai.py --mode play             # play yourself
"""

import pygame
import random
import collections
import pickle
import os
import argparse
import statistics

# ─── Constants ────────────────────────────────────────────────────────────────

GRID = 20
CELL = 28
FPS  = 15

UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)
DIRS  = [UP, DOWN, LEFT, RIGHT]

MODEL_PATH = "q_table.pkl"

# ─── Game Engine ──────────────────────────────────────────────────────────────

class SnakeGame:
    """
    Pure game logic — no pygame dependency.

    - snake   : deque of (x,y), head = snake[0]
    - step()  : takes a direction tuple (dx,dy), returns (state, reward, done)
    - _get_state() : returns 11-feature dict used by Q-agent
    """

    def __init__(self, grid=GRID, seed=None):
        self.grid = grid
        self.rng  = random.Random(seed)
        self.reset()

    def reset(self):
        mid = self.grid // 2
        self.snake     = collections.deque([(mid, mid), (mid-1, mid), (mid-2, mid)])
        self.direction = RIGHT
        self.score     = 0
        self.steps     = 0
        self.alive     = True
        self.food      = self._spawn_food()
        return self._get_state()

    def _spawn_food(self):
        empty = [(x, y) for x in range(self.grid) for y in range(self.grid)
                 if (x, y) not in self.snake]
        return self.rng.choice(empty)

    def step(self, action):
        """
        action : direction tuple — UP, DOWN, LEFT, or RIGHT.
        Returns (state_dict, reward, done).
        """
        if not self.alive:
            return self._get_state(), 0, True

        # Prevent 180° reversal
        opp = (-self.direction[0], -self.direction[1])
        if action != opp:
            self.direction = action

        head = self.snake[0]
        nx   = (head[0] + self.direction[0]) % self.grid
        ny   = (head[1] + self.direction[1]) % self.grid
        new_head = (nx, ny)

        if new_head in self.snake:
            self.alive = False
            return self._get_state(), -10, True

        self.snake.appendleft(new_head)
        self.steps += 1

        if new_head == self.food:
            self.score += 1
            reward = 10
            if len(self.snake) < self.grid * self.grid:
                self.food = self._spawn_food()
        else:
            self.snake.pop()
            old_dist = abs(head[0]-self.food[0])     + abs(head[1]-self.food[1])
            new_dist = abs(new_head[0]-self.food[0]) + abs(new_head[1]-self.food[1])
            reward   = 0.1 if new_dist < old_dist else -0.1

        return self._get_state(), reward, False

    def _get_state(self):
        """11-feature binary state used by Q-agent and for analysis."""
        head = self.snake[0]
        body = set(self.snake)

        def danger(d):
            nx = (head[0] + d[0]) % self.grid
            ny = (head[1] + d[1]) % self.grid
            return int((nx, ny) in body)

        fx, fy = self.food
        hx, hy = head

        return {
            "head": head, "food": self.food, "snake_len": len(self.snake),
            "danger_up":    danger(UP),   "danger_down":  danger(DOWN),
            "danger_left":  danger(LEFT), "danger_right": danger(RIGHT),
            "food_left":  int(fx < hx),  "food_right": int(fx > hx),
            "food_up":    int(fy < hy),  "food_down":  int(fy > hy),
        }

    def get_board(self):
        return set(self.snake), self.food, self.snake[0]


# ─── BFS Agent (Baseline) ─────────────────────────────────────────────────────

class BFSAgent:
    """
    Milestone 2 — baseline agent.

    Decision order:
      1. BFS shortest path to food.
      2. Simulate walking that path; flood-fill check reachability at the end.
         If reachable cells > half snake length → safe to go.
      3. If unsafe → tail-chase: BFS to current tail position to buy time.
      4. Last resort → pick the direction with most open cells.
    """
    name = "BFS + Safety"

    def act(self, game):
        head = game.snake[0]
        food = game.food
        body = set(game.snake)
        grid = game.grid

        # 1. Try food path with safety check
        path = self._bfs(head, food, body, grid)
        if path:
            sim = collections.deque(game.snake)
            for pos in path:
                sim.appendleft(pos)
                if pos != food:
                    sim.pop()
            if self._flood(sim[0], set(sim), grid) > len(sim) // 2:
                return self._to_dir(head, path[0], grid)   # ← convert pos→dir

        # 2. Tail-chase fallback
        tp = self._bfs(head, game.snake[-1], body, grid)
        if tp:
            return self._to_dir(head, tp[0], grid)          # ← convert pos→dir

        # 3. Max-space last resort
        best_d, best_s = None, -1
        for d in DIRS:
            nx, ny = (head[0]+d[0]) % grid, (head[1]+d[1]) % grid
            if (nx, ny) not in body:
                s = self._flood((nx, ny), body, grid)
                if s > best_s:
                    best_s, best_d = s, d
        return best_d or game.direction

    @staticmethod
    def _to_dir(head, next_pos, grid):
        """
        Convert a next-position into a direction tuple.
        Handles wrap-around: e.g. head=(0,5), next=(19,5) → LEFT.
        """
        dx = next_pos[0] - head[0]
        dy = next_pos[1] - head[1]
        # Normalise wrap-around distances
        if dx >  grid // 2: dx -= grid
        if dx < -grid // 2: dx += grid
        if dy >  grid // 2: dy -= grid
        if dy < -grid // 2: dy += grid
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        return (dx, dy)

    def _bfs(self, start, goal, body, grid):
        """BFS from start to goal, returning list of positions (not directions)."""
        q, vis = collections.deque([(start, [])]), {start}
        while q:
            pos, path = q.popleft()
            for d in DIRS:
                np_ = ((pos[0]+d[0]) % grid, (pos[1]+d[1]) % grid)
                if np_ in vis or np_ in body:
                    continue
                vis.add(np_)
                new_path = path + [np_]
                if np_ == goal:
                    return new_path
                q.append((np_, new_path))
        return None

    def _flood(self, start, body, grid):
        """Count cells reachable from start without entering body."""
        vis, q = {start}, collections.deque([start])
        while q:
            pos = q.popleft()
            for d in DIRS:
                np_ = ((pos[0]+d[0]) % grid, (pos[1]+d[1]) % grid)
                if np_ not in vis and np_ not in body:
                    vis.add(np_); q.append(np_)
        return len(vis)


# ─── Q-Learning Agent (Improved) ─────────────────────────────────────────────

class QLearningAgent:
    """
    Milestone 3 — improved agent using tabular Q-learning.

    State space  : 11 binary features → 2^11 = 2048 states
    Action space : 4 directions
    Q-table      : dict mapping (state_key, action_idx) → float value
                   Saved to disk as q_table.pkl after training.

    State key (11-tuple):
      (danger_up, danger_down, danger_left, danger_right,  ← 4 bits
       food_left, food_right, food_up, food_down,          ← 4 bits
       dir_up, dir_right, dir_down)                        ← 3 bits (LEFT implicit)

    Hyperparameters:
      alpha         : learning rate — how fast Q values update each step
      gamma         : discount factor — how much future rewards are valued
      epsilon       : starting exploration rate (1.0 = fully random)
      epsilon_decay : multiplier applied after each episode
      epsilon_min   : floor so agent always explores a tiny bit during training
    """

    name = "Q-Learning"

    ACTION_IDX = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}
    IDX_ACTION = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, alpha=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q             = {}     # Q-table: {(state_key, action_idx): float}

    # ── state encoding ───────────────────────────────────────────────────────

    def _encode(self, state, direction):
        """Map state dict + direction → compact 11-tuple key."""
        return (
            state["danger_up"],   state["danger_down"],
            state["danger_left"], state["danger_right"],
            state["food_left"],   state["food_right"],
            state["food_up"],     state["food_down"],
            int(direction == UP), int(direction == RIGHT), int(direction == DOWN),
        )

    # ── Q-table helpers ──────────────────────────────────────────────────────

    def _q(self, key, a):
        return self.q.get((key, a), 0.0)

    def _best_idx(self, key):
        vals = [self._q(key, i) for i in range(4)]
        return vals.index(max(vals))

    # ── act ──────────────────────────────────────────────────────────────────

    def act(self, game):
        """Greedy — used during watch mode and evaluation."""
        key = self._encode(game._get_state(), game.direction)
        return self.IDX_ACTION[self._best_idx(key)]

    def act_train(self, game):
        """ε-greedy — used during training."""
        if random.random() < self.epsilon:
            return random.choice(DIRS)
        return self.act(game)

    # ── Bellman update ───────────────────────────────────────────────────────

    def update(self, state, direction, action, reward, next_state, next_dir, done):
        """
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        key      = self._encode(state, direction)
        next_key = self._encode(next_state, next_dir)
        a_idx    = self.ACTION_IDX[action]
        old_q    = self._q(key, a_idx)

        target = reward if done else reward + self.gamma * max(self._q(next_key, i) for i in range(4))
        self.q[(key, a_idx)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({"q": self.q, "epsilon": self.epsilon}, f)
        print(f"  [Q] Saved → {path}  ({len(self.q)} entries)")

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            print(f"  [Q] No model at {path}. Run:  python snake_ai.py --mode train")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q       = data["q"]
        self.epsilon = self.epsilon_min
        print(f"  [Q] Loaded ← {path}  ({len(self.q)} entries)")
        return True


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(agent, n_episodes=5000, grid=GRID, max_steps=5000, save_every=1000):
    """Train Q-agent headlessly. Prints a progress line every 500 episodes."""
    print(f"\n  Training Q-agent  |  {n_episodes} eps  |  {grid}x{grid} grid")
    print(f"  α={agent.alpha}  γ={agent.gamma}  ε_start={agent.epsilon:.2f}"
          f"  ε_min={agent.epsilon_min}  decay={agent.epsilon_decay}")
    print(f"\n  {'Episode':>8}  {'Avg(500)':>10}  {'Best':>6}  {'ε':>7}  {'Q entries':>10}")
    print(f"  {'-'*50}")

    recent = collections.deque(maxlen=500)
    best   = 0

    for ep in range(1, n_episodes + 1):
        game      = SnakeGame(grid=grid)           # random seed each episode
        state     = game._get_state()
        direction = game.direction

        for _ in range(max_steps):
            action                   = agent.act_train(game)
            next_state, reward, done = game.step(action)
            next_dir                 = game.direction

            agent.update(state, direction, action, reward, next_state, next_dir, done)
            state, direction = next_state, next_dir
            if done:
                break

        agent.decay_epsilon()
        recent.append(game.score)
        if game.score > best:
            best = game.score

        if ep % 500 == 0:
            avg = sum(recent) / len(recent)
            print(f"  {ep:>8}  {avg:>10.2f}  {best:>6}  "
                  f"{agent.epsilon:>7.4f}  {len(agent.q):>10}")

        if ep % save_every == 0:
            agent.save()

    agent.save()
    print(f"\n  Training complete. Best score: {best}\n")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(agent, n_episodes=100, grid=GRID, max_steps=5000, verbose=True):
    scores, steps = [], []
    for seed in range(n_episodes):
        game = SnakeGame(grid=grid, seed=seed)
        for _ in range(max_steps):
            if not game.alive:
                break
            game.step(agent.act(game))
        scores.append(game.score)
        steps.append(game.steps)

    if verbose:
        print(f"\n{'='*47}")
        print(f"  Agent       : {agent.name}")
        print(f"  Episodes    : {n_episodes}  |  Grid: {grid}x{grid}")
        print(f"  Avg score   : {sum(scores)/len(scores):.2f}")
        print(f"  Best score  : {max(scores)}")
        print(f"  Score std   : {statistics.stdev(scores):.2f}")
        print(f"  Avg steps   : {sum(steps)/len(steps):.0f}")
        print(f"{'='*47}\n")

    return scores, steps


def compare(n_episodes=100, grid=GRID):
    bfs = BFSAgent()
    ql  = QLearningAgent()
    if not ql.load():
        return

    print(f"\n  Comparing on {n_episodes} episodes ({grid}x{grid})...")
    bs, bv = evaluate(bfs, n_episodes, grid, verbose=False)
    qs, qv = evaluate(ql,  n_episodes, grid, verbose=False)

    def delta(a, b):
        return f"{((b-a)/max(abs(a),1))*100:+.1f}%"

    print(f"\n  {'Metric':<22} {'BFS+Safety':>12} {'Q-Learning':>12} {'Δ':>8}")
    print(f"  {'-'*58}")
    ba, qa = sum(bs)/len(bs), sum(qs)/len(qs)
    print(f"  {'Avg score':<22} {ba:>12.2f} {qa:>12.2f} {delta(ba,qa):>8}")
    print(f"  {'Best score':<22} {max(bs):>12}  {max(qs):>12}  {delta(max(bs),max(qs)):>8}")
    bv_, qv_ = sum(bv)/len(bv), sum(qv)/len(qv)
    print(f"  {'Avg steps':<22} {bv_:>12.0f} {qv_:>12.0f} {delta(bv_,qv_):>8}")
    print(f"  {'Score std':<22} {statistics.stdev(bs):>12.2f} {statistics.stdev(qs):>12.2f}")
    print()


# ─── Pygame Visualizer ────────────────────────────────────────────────────────

BG        = (15,  15,  20)
GRID_LINE = (25,  25,  35)
SNAKE_H   = (80,  220, 140)
SNAKE_B   = (50,  170, 100)
FOOD_C    = (240, 80,  80)
TEXT_C    = (200, 200, 200)
DIM_C     = (100, 100, 120)

def run_game(agent=None, seed=None, cell=CELL, fps=FPS):
    pygame.init()
    W, INFO_H = GRID * cell, 60
    screen    = pygame.display.set_mode((W, W + INFO_H))
    pygame.display.set_caption("AI Snake")
    font_big  = pygame.font.SysFont("monospace", 18, bold=True)
    font_sm   = pygame.font.SysFont("monospace", 13)
    clock     = pygame.time.Clock()

    game      = SnakeGame(grid=GRID, seed=seed)
    running   = True
    paused    = False
    human_dir = RIGHT

    def draw():
        screen.fill(BG)
        for i in range(GRID + 1):
            pygame.draw.line(screen, GRID_LINE, (i*cell, 0),    (i*cell, W))
            pygame.draw.line(screen, GRID_LINE, (0,    i*cell), (W,      i*cell))

        _, food, _ = game.get_board()
        fx, fy = food
        pygame.draw.rect(screen, FOOD_C,
                         (fx*cell+3, fy*cell+3, cell-6, cell-6), border_radius=4)

        for i, (sx, sy) in enumerate(game.snake):
            col = SNAKE_H if i == 0 else SNAKE_B
            r   = 5      if i == 0 else 3
            pygame.draw.rect(screen, col,
                             (sx*cell+2, sy*cell+2, cell-4, cell-4), border_radius=r)

        pygame.draw.rect(screen, (20, 20, 28), (0, W, W, INFO_H))
        mode = agent.name if agent else "Human"
        screen.blit(font_big.render(f"Score: {game.score}", True, SNAKE_H), (12,   W+8))
        screen.blit(font_sm.render(f"Steps: {game.steps}",  True, DIM_C),   (12,   W+30))
        screen.blit(font_sm.render(f"Mode: {mode}",         True, DIM_C),   (W//2, W+8))
        screen.blit(font_sm.render("R:restart  Q:quit  SPACE:pause", True, DIM_C), (W//2, W+30))

        if not game.alive:
            ov = font_big.render("GAME OVER — press R", True, FOOD_C)
            screen.blit(ov, (W//2 - ov.get_width()//2, W//2 - 10))
        if paused:
            pv = font_big.render("PAUSED", True, TEXT_C)
            screen.blit(pv, (W//2 - pv.get_width()//2, W//2 - 10))

        pygame.display.flip()

    dir_map = {
        pygame.K_UP: UP,    pygame.K_DOWN: DOWN,
        pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
        pygame.K_w: UP, pygame.K_s: DOWN,
        pygame.K_a: LEFT, pygame.K_d: RIGHT,
    }

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    game = SnakeGame(grid=GRID, seed=seed)
                    human_dir = RIGHT
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if not agent and event.key in dir_map:
                    human_dir = dir_map[event.key]

        if game.alive and not paused:
            action = agent.act(game) if agent else human_dir
            game.step(action)

        draw()
        clock.tick(fps)

    pygame.quit()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Snake — 211401002")
    parser.add_argument("--mode",    choices=["play","ai","train","eval","compare"],
                        default="ai")
    parser.add_argument("--agent",   choices=["bfs","ql"], default="bfs")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--fps",     type=int, default=FPS)
    parser.add_argument("--eval-n",  type=int, default=100)
    parser.add_argument("--train-n", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "train":
        agent = QLearningAgent()
        train(agent, n_episodes=args.train_n)

    elif args.mode == "compare":
        compare(n_episodes=args.eval_n)

    elif args.mode == "eval":
        agent = QLearningAgent() if args.agent == "ql" else BFSAgent()
        if args.agent == "ql":
            agent.load()
        evaluate(agent, n_episodes=args.eval_n)

    elif args.mode == "play":
        run_game(agent=None, seed=args.seed, fps=args.fps)

    else:  # ai (default)
        agent = QLearningAgent() if args.agent == "ql" else BFSAgent()
        if args.agent == "ql":
            agent.load()
        run_game(agent=agent, seed=args.seed, fps=args.fps)