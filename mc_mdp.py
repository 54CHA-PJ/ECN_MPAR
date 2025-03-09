# mc_mdp.py
import sys
from colorama import Fore, Style

from antlr4 import *
from gramListener import gramListener
from gramParser import gramParser

class gramPrintListener(gramListener):
    """ 
    Builds an internal model (states, actions, transitions, etc.)
    and offers multiple probability analysis methods (symbolic, iterative, statistical).
    """
    def __init__(self):
        super().__init__()
        self.states = []           
        self.actions = []          
        self.transitions = []      
        self.rewards = {}          
        self.warning_state_list = []

    # --------------------
    # GRAMMAR HOOKS

    def enterDefstates(self, ctx: gramParser.DefstatesContext):
        for sctx in ctx.statedef():
            state_name = sctx.ID().getText()
            if sctx.INT():
                rew = int(sctx.INT().getText())
            else:
                rew = 0
                self.warning_state_list.append(state_name)
            self.states.append(state_name)
            self.rewards[state_name] = rew

    def enterDefactions(self, ctx: gramParser.DefactionsContext):
        self.actions = [str(x) for x in ctx.ID()]

    def enterTransact(self, ctx: gramParser.TransactContext):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MDP", dep, act, ids, weights))

    def enterTransnoact(self, ctx: gramParser.TransnoactContext):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MC", dep, None, ids, weights))

    # --------------------
    # VALIDATION

    def validate(self):
        valid = True
        # Warnings for states without explicit reward
        for warn_state in self.warning_state_list:
            print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} State {Fore.YELLOW}{warn_state}{Style.RESET_ALL} has no reward (default 0).")

        # Must have S0 as an initial state
        if "S0" not in self.states:
            print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} No initial state 'S0' found.")
            valid = False

        # Check transitions
        for t_type, dep, act, dests, weights in self.transitions:
            # Action declared?
            if act and act not in self.actions:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Action {Fore.YELLOW}{act}{Style.RESET_ALL} not declared.")
            # Negative weights?
            if any(w < 0 for w in weights):
                print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Negative weight found in transition {dep} -> {dests}.")
                valid = False
            # Mixed MDP/MC from the same departure?
            if t_type == "MC":
                for other in self.transitions:
                    if other[0] == "MDP" and other[1] == dep:
                        print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Mixed MC+MDP on state {dep}.")
                        valid = False
            # Destination states unknown?
            for d in dests:
                if d not in self.states:
                    print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Destination state {Fore.YELLOW}{d}{Style.RESET_ALL} unknown.")
        return valid

    def check(self, filename):
        print(Fore.LIGHTBLUE_EX + f"\nModel: {filename}" + Style.RESET_ALL)
        if not self.validate():
            print(Fore.LIGHTRED_EX + "Model is not valid! Quitting..." + Style.RESET_ALL)
            sys.exit(1)
        print(Fore.LIGHTGREEN_EX + "Model is valid." + Style.RESET_ALL)
        self.describe()

    def describe(self):
        print(Fore.LIGHTBLUE_EX + "\n------------------ Model Description ----------------" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "States  :" + Style.RESET_ALL, self.states)
        print(Fore.LIGHTBLUE_EX + "Rewards :" + Style.RESET_ALL, self.rewards)
        print(Fore.LIGHTBLUE_EX + "Actions :" + Style.RESET_ALL, self.actions)
        print(Fore.LIGHTBLUE_EX + "Transitions :" + Style.RESET_ALL)
        for t in self.transitions:
            print("   ", t)
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)

    # --------------------
    # MAIN ANALYSIS

    def get_matrix(self):
        """ Returns a (rows, desc, state_list) describing transitions as numeric probabilities. """
        import numpy as np
        state_list = self.states[:]
        rows = []
        desc = []
        for idx, (t_type, dep, act, dests, weights) in enumerate(self.transitions):
            total = sum(weights)
            row = np.zeros(len(state_list))
            for d, w in zip(dests, weights):
                prob = w/total if total else 0
                row[state_list.index(d)] = prob
            label = f"[{idx}] ({dep}, {act})" if t_type == "MDP" else f"[{idx}]    ({dep})"
            desc.append(label)
            rows.append(row)
        return rows, desc, state_list

    def get_state_analysis(self, initial_win_states=["S0"]):
        """ 
        Identify states as win/lose/incertitude.
        A 'win' if all transitions lead to already known win states.
        A 'lose' if no transitions or all self-loops.
        Others incertitude.
        """
        win_states = initial_win_states[:]
        changed = True
        while changed:
            changed = False
            for s in self.states:
                if s in win_states:
                    continue
                out_t = [t for t in self.transitions if t[1] == s]
                if not out_t:
                    # no transitions => can't be an immediate winner
                    continue

                # Check MC transitions => all destinations in win_states
                mc_ok = all(all(d in win_states for d in t[3]) for t in out_t if t[0] == "MC")
                if not mc_ok:
                    continue

                # Check MDP transitions => for each distinct action, all destinations in win_states
                mdp_t = [t for t in out_t if t[0] == "MDP"]
                if mdp_t:
                    actions = set(tt[2] for tt in mdp_t)
                    mdp_ok = True
                    for a in actions:
                        same_action = [tt for tt in mdp_t if tt[2] == a]
                        # if any dest not in win => fail
                        if any(any(dest not in win_states for dest in trans[3]) for trans in same_action):
                            mdp_ok = False
                            break
                    if not mdp_ok:
                        continue

                win_states.append(s)
                changed = True

        lose_states = []
        for s in self.states:
            if s in win_states:
                continue
            out_t = [t for t in self.transitions if t[1] == s]
            if not out_t or all(t[3] == [s] for t in out_t):
                lose_states.append(s)

        incertitude = [s for s in self.states if s not in win_states and s not in lose_states]
        return win_states, lose_states, incertitude

    # --------------------
    # MULTIPLE PROBA METHODS

    def proba_symbolic(self, win_set, doubt_set):
        """ Symbolic approach for MC (already implemented). """
        import numpy as np
        n = len(doubt_set)
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i, s in enumerate(doubt_set):
            out_t = [t for t in self.transitions if t[1] == s]
            if not out_t:
                continue
            # We'll pick the first transition from s for MC
            trans_type, dep, act, dests, weights = out_t[0]
            total = sum(weights)
            for d, w in zip(dests, weights):
                prob = w/total if total else 0
                if d in doubt_set:
                    j = doubt_set.index(d)
                    A[i, j] += prob
                elif d in win_set:
                    b[i] += prob
        I_minus_A = np.eye(n) - A
        x = np.linalg.solve(I_minus_A, b)
        return x

    def proba_iterative(self, win_set, doubt_set):
        """ Iterative approach (placeholder).
            This should compute the same probabilities by iteration. """
        import numpy as np
        print("[Iterative] Not implemented yet. Returning dummy zeros.")
        return np.zeros(len(doubt_set))

    def proba_statistical(self, win_set, doubt_set, trials=10000):
        """ Statistical approach (placeholder).
            E.g. random simulation to estimate probabilities. """
        import numpy as np
        print("[Statistical] Not implemented yet. Returning dummy zeros.")
        return np.zeros(len(doubt_set))
