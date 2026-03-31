import numpy as np

class Node:
    def __init__(self, state, h, prior, cost=0, action_mol=None, fmove=0, reaction=None, template=None, parent=None, cpuct=4.0):
        self.state = state
        self.cost = cost
        self.h = h
        self.prior = prior
        self.visited_time = 0
        self.is_expanded = False
        self.template = template
        self.action_mol = action_mol
        self.fmove = fmove
        self.reaction = reaction
        self.parent = parent
        self.cpuct = cpuct
        self.children = []
        self.child_illegal = np.array([])
        if parent is not None:
            self.g = self.parent.g + cost
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.g = 0
            self.depth = 0
        self.f = self.g + self.h
        self.f_mean_path = []
    
    def partial_route(self):
        route = []
        node = self
        while node.parent is not None:
            route.append(node.reaction)
            node = node.parent
        route.reverse()
        return route

    def child_N(self):
        N = [child.visited_time for child in self.children]
        return np.array(N)

    def child_p(self):
        prior = [child.prior for child in self.children]
        return np.array(prior)

    def child_U(self):
        child_Ns = self.child_N() + 1
        prior = self.child_p()
        child_Us = self.cpuct * np.sqrt(self.visited_time) * prior / child_Ns
        return child_Us

    def child_Q(self, min_max_stats):
        child_Qs = []
        for child in self.children:
            if len(child.f_mean_path) == 0:
                child_Qs.append(0.0)
            else:
                child_Qs.append(1 - np.mean(min_max_stats.normalize(child.f_mean_path)))
        return np.array(child_Qs)

    def select_child(self, all_evaluate, min_max_stats, output):
        action_score = self.child_Q(min_max_stats) + self.child_U() - self.child_illegal       
        for i, child in enumerate(self.children):
            action_score[i] = all_evaluate(action_score[i], child, output, force_compute=False)
        # print("MCTS Constraint-Adjusted Action Scores: ", action_score)
        # choose highest-scoring child that's not illegal; if all are illegal, fall back to the best (highest-score) move
        order = np.argsort(-action_score)  # indices sorted by descending score
        original_best = int(np.argmax(action_score))
        best_move = original_best
        for idx in order:
            if self.child_illegal[idx] <= 0:
                best_move = int(idx)
                break
            else:
                print(f"DEBUGGING>>> Move {int(idx)} is illegal (score={action_score[idx]}).")
        else:
            # no legal moves found; keep original best
            print("DEBUGGING>>> All candidate moves are illegal. Returning best (illegal) move.")
            return original_best
        return best_move

class MinMaxStats(object):
    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value) -> float:
        if self.maximum > self.minimum:
            return (np.array(value) - self.minimum) / (self.maximum - self.minimum)
        return value
