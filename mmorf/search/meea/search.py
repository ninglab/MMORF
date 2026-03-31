from .graph import Node, MinMaxStats
from .search_utils import batch_smiles_to_fp, is_feasible
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from ..interrupt_manager import InterruptManager
from .value_function_manager import ValueFunctionManager
from mmorf.utils.canonicalize import canonicalize
from mmorf.utils.template_matching import reaction_from_smiles
from rdkit.Chem import rdChemReactions

class MCTS_A(object):
    def __init__(self, target_mol, known_mols, value_model, expand_fn, device, simulations, cpuct, vfm : ValueFunctionManager, interrupt_mgr : InterruptManager, output, restrictions=None, task_context=None, max_depth=-1, feasible=False, fail_on_root_illegal=False, enable_further_expansion=True, starting_iterations=1, auto_further_expand=False, meea_original=False):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.auto_further_expand = auto_further_expand
        self.expand_fn = expand_fn
        self.value_model = value_model
        self.force_feasible = feasible
        self.route = None
        self.task_context = task_context if task_context is not None else ""
        self.fail_on_root_illegal = fail_on_root_illegal
        self.output = output
        self.vfm = vfm
        self.interrupt_mgr = interrupt_mgr
        self.is_feasible = is_feasible
        self.restricted_nodes = []
        self.device = device
        self.cpuct = cpuct
        root_value = self.orig_value_fn(self.value_model, [target_mol], self.device)
        self.root = Node([target_mol], root_value, prior=1.0, cpuct=self.cpuct)
        self.feedback = ""
        self.enable_further_expansion = enable_further_expansion
        self.rejected_routes = []
        self.open = [self.root]
        self.visited_policy = {}
        self.visited_state = []
        self.next_node = None
        self.continue_iterations = starting_iterations
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)
        self.meea_original = meea_original
        # Keep both keys for depth to remain compatible with different callers
        if restrictions:
            self.restrictions = restrictions
        else:
            self.restrictions = {"specific_reactions": [], "molecules": [], "reaction_templates": [], "max_depth": max_depth}
        self.opening_size = simulations
        self.iterations = 0
        self.non_expanding_iterations = 0
    
    def revise_restrictions(self):
        unrestricted = []
        for node in self.restricted_nodes:
            if self.is_restricted(node.reaction) or (self.force_feasible and not is_feasible(node.reaction)):
                continue # still restricted
            elif node.parent is not None:
                # no longer restricted, unrestrict in tree
                unrestricted.append(node)
                node.parent.child_illegal[node.fmove] = 0
                back_check_node = node.parent
                while back_check_node != None and back_check_node.parent != None:
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 0
                    back_check_node = back_check_node.parent
        for node in unrestricted:
            self.restricted_nodes.remove(node)

    def all_evaluate(self, *args, **kwargs):
        return self.vfm.all_evaluate(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.vfm.evaluate(*args, **kwargs)

    def orig_value_fn(self, model, mols, device):
        num_mols = len(mols)
        fps = batch_smiles_to_fp(mols, fp_dim=2048).reshape(num_mols, -1)
        index = len(fps)
        if len(fps) <= 5:
            mask = np.ones(5)
            mask[index:] = 0
            fps_input = np.zeros((5, 2048))
            fps_input[:index, :] = fps
        else:
            mask = np.ones(len(fps))
            fps_input = fps
        fps = torch.FloatTensor([fps_input.astype(np.float32)]).to(device)
        mask = torch.FloatTensor([mask.astype(np.float32)]).to(device)
        v = model(fps, mask).cpu().data.numpy()
        return v[0][0]

    def select_a_leaf(self):
        current = self.root
        if self.meea_original:
            print("DEBUGGING>>> ORIGINAL MEEA* LEAF SELECTION")
            while True:
                current.visited_time += 1
                if not current.is_expanded:
                    return current
                best_move = current.select_child(self.all_evaluate, self.min_max_stats, self.output)
                current = current.children[best_move]
        else:
            while True:
                if self.restrictions.get("max_depth", -1) is None:
                    self.restrictions["max_depth"] = -1
                current.visited_time += 1
                if not current.is_expanded:
                    return current
                if current.depth >= self.restrictions.get("max_depth", -1) and self.restrictions.get("max_depth", -1) > 0:
                    print("DEBUGGING>>> Reached max depth at node with molecule: ", current.state)
                    # if self.enable_further_expansion: print("Returning parent")
                    # else: print("Returning None")
                    return None
                    # return current.parent if self.enable_further_expansion else None
                best_move = current.select_child(self.all_evaluate, self.min_max_stats, self.output)
                # if (current.child_illegal[best_move] > 0) and self.enable_further_expansion:
                #     print("All child nodes are illegal, stopping selection at last legal node.")
                #     return current
                if current.child_illegal[best_move] > 0:
                    print("All child nodes are illegal, returning None.")
                    return None
                current = current.children[best_move]

    def select(self):
        openings = []
        retries = 1
        if self.fail_on_root_illegal and np.all(self.root.child_illegal > 0) and len(self.root.children) > 0:
            print("DEBUGGING>>> Root node fully restricted, cannot select any node.")
            exit(2)
        while len(openings) == 0 and retries > 0:
            retries-=1
            openings = [self.select_a_leaf() for _ in range(self.opening_size)]
            openings = [opening for opening in openings if opening is not None]
        if len(openings) == 0:
            print("DEBUGGING>>> No valid openings")
            return None
        stats = [opening.f for opening in openings]
        if self.meea_original:
            # this skips all normalization / transformation of floating point heuristic values in an attempt to reproduce the original MEEA* results
            index = np.argmin(stats)
            return openings[index]
        stats = [float(s) for s in stats] # ensure all floats for argmin
        # print("pre-minmax stats: ", stats)
        if max(stats) == min(stats):
            stats = [1.0 for s in stats]
        else:
            stats = [(float(s) - min(stats)) / (max(stats) - min(stats)) for s in stats] # normalize to [0, 1]
        # print("pre-constraint scores: ", stats)
        const = [self.all_evaluate(1-stats[i], opening, self.output, force_compute=True) for i, opening in enumerate(openings)]
        context = self.task_context + "\n" + self.feedback
        if len(openings) == 0:
            context += f"\nNo valid expansions are available."
        if self.interrupt_mgr.emit("simulation_complete", value_fn_scores=const, original_scores=stats, nodes=openings, search=self, context=context):
            # returns True when no change to the scores or nodes or value function or restrictions.
            if self.next_node is not None:
                return self.next_node
            index = np.argmax(const) # flip to argmax so we can use a consistent scoring function to the value fn.
            myconst = const.copy()
            while openings[index].is_expanded: # do not select already expanded nodes
                myconst[index] = -float('inf')
                if max(myconst) == -float('inf'):
                    return None
                index = np.argmax(myconst)
            return openings[index]
        else:
            self.non_expanding_iterations += 1
            return None
        
    def is_restricted(self, reaction):
        # Check molecules: canonicalize both sides for robust matching
        state_mols = set()
        for m in reaction.replace(">>",".").split("."):
            try:
                state_mols.add(canonicalize(m))
            except Exception:
                pass
        restricted_mols = set()
        for m in self.restrictions.get("molecules", []):
            try:
                restricted_mols.add(canonicalize(m))
            except Exception:
                restricted_mols.add(m)
        if len(state_mols.intersection(restricted_mols)) > 0:
            return True

        # Specific reactions: compare products and canonicalized reactant sets (see pruning.get_restricted_routes)
        for rxn in self.restrictions.get("specific_reactions", []):
            try:
                if reaction and ">>" in reaction and ">>" in rxn:
                    if reaction.split(">>")[1].strip() == rxn.split(">>")[1].strip():
                        reactants = canonicalize(reaction.split(">>")[0]).split('.') if reaction.split(">>")[0].strip() != '' else []
                        rxn_reactants = canonicalize(rxn.split(">>")[0]).split('.') if rxn.split(">>")[0].strip() != '' else []
                        if set(reactants) == set(rxn_reactants):
                            return True
            except Exception:
                continue

        # Reaction templates: try SMARTS matching using RDKit reaction substructure matching
        for smarts in self.restrictions.get("reaction_templates", []):
            try:
                patt = AllChem.ReactionFromSmarts(smarts)
                patt.Initialize()
                if reaction and ">>" in reaction:
                    try:
                        rxn = reaction_from_smiles(reaction.split(">>")[0], reaction.split(">>")[1])
                        rxn.Initialize()
                        if rdChemReactions.HasReactionSubstructMatch(rxn, patt):
                            return True
                    except Exception:
                        continue
            except Exception:
                continue
        return False
    
    def further_expand(self, node, reactions):
        # print("Error: further_expand called but disabled.")
        # return False, None
        try:
            assert node.is_expanded, "Node not yet expanded."
            assert self.enable_further_expansion, "Further expansion disabled."
        except AssertionError as e:
            print("ERROR in further_expand: ", str(e))
            return False, None
        expanded_policy = {'reactants': [], 'scores': [], 'template': []}
        for reaction in reactions:
            reactant = reaction.split(">>")[0] if ">>" in reaction else reaction
            expanded_policy['reactants'].append(reactant)
            expanded_policy['scores'].append(1.0) # uniform score for user-specified reactions
            expanded_policy['template'].append('') # template unknown for user-specified reactions
            node.child_illegal = np.append(node.child_illegal, 0)
        self.visited_policy[node.state[0]] = self.visited_policy.get(node.state[0], None)
        if self.visited_policy[node.state[0]] is None:
            self.visited_policy[node.state[0]] = {'reactants': [], 'scores': [], 'template': []}
        self.visited_policy[node.state[0]]["reactants"] += expanded_policy['reactants']
        self.visited_policy[node.state[0]]["scores"] += expanded_policy['scores']
        self.visited_policy[node.state[0]]["template"] += expanded_policy['template']
        if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
            print("DEBUGGING>>> Further expanding node with molecule: ", node.state[0], " with ", len(expanded_policy["scores"]), " reactions.")
            for i in range(len(expanded_policy['scores'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                reactant = reactant + node.state[1:]
                reactant = sorted(list(set(reactant)))
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + node.state[0]
                priors = np.array([1.0 / len(expanded_policy['scores'])] * len(expanded_policy['scores']))
                if len(reactant) == 0:
                    child = Node([], 0, cost=cost, prior=priors[i], action_mol=node.state[0], reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    if not (self.force_feasible and not is_feasible(reaction)):
                        self.interrupt_mgr.emit("all_routes", route=self.vis_synthetic_path(child)[0][1:])
                    if self.is_restricted(reaction) or (self.force_feasible and not is_feasible(reaction)):
                        self.restricted_nodes.append(child)
                        node.child_illegal[child.fmove] = 1e6
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                            back_check_node = back_check_node.parent
                    else:
                        if self.interrupt_mgr.emit("route_found", node=child, route=self.vis_synthetic_path(child)[0][1:], search=self):
                            return True, self.route
                        else:
                            node.child_illegal[child.fmove] = 1e6
                            back_check_node = node
                            while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                                back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                                back_check_node = back_check_node.parent
                else:
                    h = self.orig_value_fn(self.value_model, reactant, self.device)
                    child = Node(reactant, h, cost=cost, prior=priors[i], action_mol=node.state[0], reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    if self.is_restricted(child.reaction) or (self.force_feasible and not is_feasible(reaction)):
                        print("DEBUGGING>>> Reaction restricted or infeasible: ", child.reaction)
                        self.restricted_nodes.append(child)
                        node.child_illegal[child.fmove] = 1e6
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                            back_check_node = back_check_node.parent
                    if ('.'.join(reactant) in self.visited_state) or (Chem.MolFromSmiles(".".join(reactant)) is None):
                        node.child_illegal[child.fmove] = 1e6
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                            back_check_node = back_check_node.parent
        else:
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1e6
                back_check_node = node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                    back_check_node = back_check_node.parent
        return False, None

    def is_predecessor_restricted(self, node):
        if node.parent is None:
            return False
        else:
            if node.parent.reaction:
                if self.is_restricted(node.parent.reaction):
                    return True
            return self.is_predecessor_restricted(node.parent)

    def expand(self, node):
        if len(node.state) == 0:
            return False, None
        if node.is_expanded:
            self.further_reactions = []
            if self.enable_further_expansion:
                if self.interrupt_mgr.emit("expansion_coordinator", node.state[0], node=node, search=self):
                    return self.further_expand(node, self.further_reactions)
            else:
                print("Further expansion disabled, skipping.")
                return False, None
        node.is_expanded = True
        expanded_mol_index = 0
        expanded_mol = node.state[expanded_mol_index]
        if expanded_mol in self.visited_policy.keys():
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.expand_fn.run(expanded_mol, topk=50)
            self.iterations += 1
            if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
                self.visited_policy[expanded_mol] = expanded_policy.copy()
            else:
                self.visited_policy[expanded_mol] = None
        if expanded_policy is not None and (len(expanded_policy['scores']) > 0):
            print("DEBUGGING>>> Expanding node with molecule: ", expanded_mol, " with ", len(expanded_policy["scores"]), " reactions.")
            node.child_illegal = np.array([0] * len(expanded_policy['scores']))
            for i in range(len(expanded_policy['scores'])):
                reactant = [r for r in expanded_policy['reactants'][i].split('.') if r not in self.known_mols]
                reactant = reactant + node.state[: expanded_mol_index] + node.state[expanded_mol_index + 1:]
                reactant = sorted(list(set(reactant)))
                cost = - np.log(np.clip(expanded_policy['scores'][i], 1e-3, 1.0))
                template = expanded_policy['template'][i]
                reaction = expanded_policy['reactants'][i] + '>>' + expanded_mol
                priors = np.array([1.0 / len(expanded_policy['scores'])] * len(expanded_policy['scores']))
                if len(reactant) == 0:
                    # print("DEBUGGING>>> No reactant.")
                    child = Node([], 0, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    if not (self.force_feasible and not is_feasible(reaction)):
                        self.interrupt_mgr.emit("all_routes", route=self.vis_synthetic_path(child)[0][1:])
                    if self.is_restricted(reaction) or self.is_predecessor_restricted(child) or (self.force_feasible and not is_feasible(reaction)) or (child.depth > self.restrictions.get("max_depth", -1) > 0):
                        self.restricted_nodes.append(child)
                        # print("DEBUGGING>>> Node depth exceeds max depth, skipping expansion.")
                        node.child_illegal[child.fmove] = 1e6
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                            back_check_node = back_check_node.parent
                    else:
                        print("DEBUGGING>>> Potential route found, checking with agent.", self.restrictions)
                        if self.interrupt_mgr.emit("route_found", node=child, route=self.vis_synthetic_path(child)[0][1:], search=self):
                            return True, self.route
                        else:
                            print("DEBUGGING>>> Route rejected by Agent.", self.restrictions)
                            node.child_illegal[child.fmove] = 1e6
                            back_check_node = node
                            while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                                back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                                back_check_node = back_check_node.parent
                else:
                    # print("DEBUGGING>>> %d reactants." % len(reactant))
                    h = self.orig_value_fn(self.value_model, reactant, self.device)
                    child = Node(reactant, h, cost=cost, prior=priors[i], action_mol=expanded_mol, reaction=reaction, fmove=len(node.children), template=template, parent=node, cpuct=self.cpuct)
                    if self.is_restricted(child.reaction) or self.is_predecessor_restricted(child) or (self.force_feasible and not is_feasible(reaction)) or (child.depth > self.restrictions.get("max_depth", -1) > 0):
                        print("DEBUGGING>>> Reaction restricted or infeasible: ", child.reaction)
                        # print("DEBUGGING>>> Node depth exceeds max depth, skipping expansion.")
                        self.restricted_nodes.append(child)
                        node.child_illegal[child.fmove] = 1e6
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                            back_check_node = back_check_node.parent
                    if ('.'.join(reactant) in self.visited_state) or (Chem.MolFromSmiles(".".join(reactant)) is None):
                        node.child_illegal[child.fmove] = 1e6
                        back_check_node = node
                        while back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                            back_check_node = back_check_node.parent
        else:
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1e6
                back_check_node = node.parent
                while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                    back_check_node = back_check_node.parent
        if self.auto_further_expand and self.enable_further_expansion:
            self.further_reactions = []
            if self.interrupt_mgr.emit("expansion_coordinator", node.state[0], node=node, search=self):
                # return self.further_expand(node, self.further_reactions)
                pass
        return False, None

    def update(self, node):
        stat = node.f
        self.min_max_stats.update(stat)
        current = node
        while current is not None:
            current.f_mean_path.append(stat)
            current = current.parent

    def search(self, times, non_expanding_limit="shared"):
        self.times = times
        success, route = False, None
        iters = lambda: self.iterations
        if non_expanding_limit == "shared":
            iters = lambda: self.non_expanding_iterations + self.iterations
        while iters() < times and not success:
            if self.root.is_expanded and (np.all(self.root.child_illegal > 0) or len(self.root.child_illegal) == 0):
                print("DEBUGGING>>> Route is fully restricted or no reactions are feasible.")
                self.feedback = "The entire search space consists of infeasible reactions or restricted reactions.  Consider relaxing restrictions, if any, or continuing with by re-expanding a route based on its last feasible/unrestricted intermediate."
            expand_node = self.select()
            if expand_node is None:
                print("DEBUGGING>>> Expansion skipped by agent.")
                continue
                # print("DEBUGGING>>> No valid node to expand.")
                # return False, None, iters()
            counter = 0
            if self.iterations % 10 == 0:
                print("TIME", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("DEBUGGING>>> Iteration: %d, Open size: %d, Visited states: %d" % (self.iterations, len(self.open), len(self.visited_state)))
            # print("DEBUGGING>>> Selected node: ", expand_node.state)
            if '.'.join(expand_node.state) in self.visited_state:
                print("DEBUGGING>>> Already visited.")
                if self.enable_further_expansion:
                    success, route = self.further_expand(expand_node, [])
                    self.update(expand_node)
                    # print("DEBUGGING>>> Further expansion disabled, skipping.")
                elif expand_node.parent is not None:
                    # if expand_node.parent is not None:
                    expand_node.parent.child_illegal[expand_node.fmove] = 1e6
                    back_check_node = expand_node.parent
                    while back_check_node != None and back_check_node.parent != None and np.all(back_check_node.child_illegal > 0):
                        back_check_node.parent.child_illegal[back_check_node.fmove] = 1e6
                        back_check_node = back_check_node.parent
            else:
                self.visited_state.append('.'.join(expand_node.state))
                success, route = self.expand(expand_node)
                self.update(expand_node)
            if self.visited_policy[self.target_mol] is None:
                print("DEBUGGING>>> No visited policy for target molecule.")
                print("TIME", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("DEBUGGING>>> Iteration: %d, Open size: %d, Visited states: %d" % (self.iterations, len(self.open), len(self.visited_state)))
                return False, None, iters()
            else:
                pass
            # print("DEBUGGING>>> Visited policy for target molecule: ", self.visited_policy[self.target_mol])
        print("TIME", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("DEBUGGING>>> Iteration: %d, Open size: %d, Visited states: %d" % (self.iterations, len(self.open), len(self.visited_state)))
        return success, route, self.iterations

    def vis_synthetic_path(self, node):
        # navigates path of states from solved state to root.
        # states include all open moleucules at each step, allowing for full route reconstruction.
        if node is None:
            return [], []
        reaction_path = []
        template_path = []
        current = node
        while current is not None:
            reaction_path.append(current.reaction)
            template_path.append(current.template)
            current = current.parent
        return reaction_path[::-1], template_path[::-1]
