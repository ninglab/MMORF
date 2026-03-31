from .search import MCTS_A
from .search_utils import prepare_value, prepare_expand, prepare_starting_molecules, is_feasible
import os, json, time
from ...utils.logger import Logger
from .value_function_manager import ValueFunctionManager
from ..base import BaseSearch
from ..interrupt_manager import InterruptManager

class MEEA(BaseSearch):
    def __init__(self, stock=None, device="cuda:0", value_fun_string="Synth()",
                 cpuct=4.0, restrictions=None, output=None, feasible=False, fail_on_root_illegal=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.known_mols = stock
        self.fail_on_root_illegal = fail_on_root_illegal
        if self.known_mols is None:
            self.known_mols = prepare_starting_molecules()
        self.device = device
        self.cpuct = cpuct
        self.output = output
        exp_model_path = os.path.dirname(__file__) + '/../../saved_model/policy_model.ckpt'
        value_model_f = os.path.dirname(__file__) + '/../../saved_model/value_pc.pt'
        self.value_model = prepare_value(value_model_f, device)
        self.expand_fn = prepare_expand(exp_model_path, device)
        self.vfm = ValueFunctionManager(value_fun_string)
        self.restrictions = restrictions
        self.start_time = None
        self.setting = None
        self.interrupt_mgr = InterruptManager(parallel=kwargs.get("parallel_interrupts", 0))
        if restrictions is None:
            self.restrictions = []
        if output is None:
            self.output = "output"
        self.feasible = feasible
        self.enable_further_expansion = kwargs.get("enable_further_expansion", False)

    def __call__(self, mol, context=None, k=5, iteration_limit=500, start_iter=50, auto_further_expand=False, **kwargs):
        self.simulations = k
        self.player = MCTS_A(mol, self.known_mols, self.value_model, self.expand_fn, self.device, self.simulations, self.cpuct, self.vfm, self.interrupt_mgr, self.output, restrictions=self.restrictions, feasible=self.feasible, task_context=context, starting_iterations=start_iter, auto_further_expand=auto_further_expand, enable_further_expansion=self.enable_further_expansion, fail_on_root_illegal=self.fail_on_root_illegal, meea_original=kwargs.get("meea_original", False))
        self.player.start_time = self.start_time if self.start_time is not None else time.time()
        self.player.setting = self.setting
        success, route, count = self.player.search(iteration_limit)
        if success:
            Logger.log(f"Route found with {count} nodes expanded.")
        return route
