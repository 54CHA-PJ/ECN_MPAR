# mc_mdp.py
import sys
from colorama import Fore, Style

from antlr4 import *
from gramListener import gramListener
from gramParser import gramParser

class gramPrintListener(gramListener):
    """ 
    Fields:
      - states: list of states (strings) in the order they are defined.
      - actions: list of actions (strings) in the order they are defined.
      - rewards: dictionary mapping state -> reward (default 0 if not provided)
      - transitions:
          - type: "MDP" or "MC"
          - dep: departure state
          - act: action (if MDP)
          - dest_states: list of destination states
          - weights: list of weights
    """
    def __init__(self):
        super().__init__()
        self.states = []            # List of states in order of appearance
        self.actions = []           # List of actions in order of appearance
        self.transitions = []       # List of transitions (type, dep, act, dest_states, weights)
        self.rewards = {}           # Dictionary of rewards for each state

    def enterDefstates(self, ctx: gramParser.DefstatesContext):
        """ Capture states and rewards """
        self.warning_state_list = []    # States with no reward defined
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
        """ Capture actions """
        self.actions = [str(x) for x in ctx.ID()]

    def enterTransact(self, ctx: gramParser.TransactContext):
        """ Capture MDP transitions """
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MDP", dep, act, ids, weights))

    def enterTransnoact(self, ctx: gramParser.TransnoactContext):
        """ Capture MC transitions """
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MC", dep, None, ids, weights))

    def describe(self):
        """ Print a summary of the model """
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "States: " + Style.RESET_ALL, self.states)
        print(Fore.LIGHTBLUE_EX + "Rewards: " + Style.RESET_ALL, self.rewards)
        print(Fore.LIGHTBLUE_EX + "Actions: " + Style.RESET_ALL, self.actions)
        print(Fore.LIGHTBLUE_EX + "Transitions:" + Style.RESET_ALL)
        for t in self.transitions:
            print(" -", t)
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)

    def validate(self):
        """ Check if the model is valid """
        valid = True
        # Check if all states have a reward
        for warn_state in self.warning_state_list:
            print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} State {Fore.YELLOW}{warn_state}{Style.RESET_ALL} has no reward defined (default 0).")
        # Check if S0 is defined
        if "S0" not in self.states:
            print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} No initial state S0 is defined!")
            valid = False
        # Check if all transitions are valid
        for t in self.transitions:
            trans_type, dep, act, dests, weights = t
            # Check if action is valid
            if act and act not in self.actions:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Action {Fore.YELLOW}{act}{Style.RESET_ALL} not declared! Adding to actions list.")
                self.actions.append(act)
            # Check if departure state is valid
            if dep not in self.states:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Departure state {Fore.YELLOW}{dep}{Style.RESET_ALL} unknown. Adding to states list.")
                self.states.append(dep)
            # Check dests states are valid
            for d in dests:
                if d not in self.states:
                    print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Destination state {Fore.YELLOW}{d}{Style.RESET_ALL} unknown. Adding to states list.")
                    self.states.append(d)
            # Check for negative weights
            if any(w < 0 for w in weights):
                print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Negative weight in transition {t}")
                valid = False
            # Mixed MDP/MC for the same departure
            if trans_type == "MC":
                # Check if any MDP shares the same departure
                for other in self.transitions:
                    if other[0] == "MDP" and other[1] == dep:
                        print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Can't combine MC + MDP transitions for state {dep}")
                        valid = False
        # If no error, it is valid (Even if there are warnings)
        return valid

    def check(self, filename):
        print(Fore.LIGHTBLUE_EX + f"\nModel: {filename}" + Style.RESET_ALL)
        if not self.validate():
            print(Fore.LIGHTRED_EX + "Model is not valid! Quitting..." + Style.RESET_ALL)
            sys.exit(1)
        else:
            print(Fore.LIGHTGREEN_EX + "Model is valid." + Style.RESET_ALL)
        self.describe()

    def get_matrix(self):
        """
        Build transition matrix rows, one row per transition, 
        plus a descriptor and state_list (columns).
        """
        import numpy as np
        state_list = self.states[:]
        rows = []
        desc = []
        for idx, (t_type, dep, act, dests, weights) in enumerate(self.transitions):
            total = sum(weights)
            row = np.zeros(len(state_list))
            for dest, w in zip(dests, weights):
                prob = w / total if total else 0
                row[state_list.index(dest)] = prob
            label = f"[{idx}] ({dep}, {act})" if t_type == "MDP" else f"[{idx}] ({dep})"
            desc.append(label)
            rows.append(row)
        return rows, desc, state_list

    def get_state_analysis(self, initial_win_states=["S0"]):
        """
        Identify (win, lose, incertitude) states.
        - Win: all transitions lead to already known win states
        - Lose: no transitions OR all self loops
        - Others: incertitude
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
                    # no transitions => can't be immediate winning
                    continue
                # Check MC transitions => all destinations in win_states
                mc_ok = all(all(dest in win_states for dest in t[3]) for t in out_t if t[0] == "MC")
                if not mc_ok:
                    continue
                # Check MDP transitions => for each distinct action, all destinations in win_states
                mdp_t = [tt for tt in out_t if tt[0] == "MDP"]
                if mdp_t:
                    actions = set(tt[2] for tt in mdp_t)
                    mdp_ok = True
                    for a in actions:
                        same_action = [tt for tt in mdp_t if tt[2] == a]
                        # If any dest is not in win_states => not OK
                        if any(any(d not in win_states for d in trans[3]) for trans in same_action):
                            mdp_ok = False
                            break
                    if not mdp_ok:
                        continue
                # If we get here => s is a new win
                win_states.append(s)
                changed = True

        # lose states
        lose_states = []
        for s in self.states:
            if s in win_states:
                continue
            out_t = [t for t in self.transitions if t[1] == s]
            if not out_t or all(t[3] == [s] for t in out_t):
                lose_states.append(s)

        incertitude = [s for s in self.states if s not in win_states and s not in lose_states]
        return win_states, lose_states, incertitude

    def calcul_proba_MC(self, win_set, doubt_set, transitions):
        """Compute probabilities of eventually reaching a win state for a MC among the doubt_set states."""
        import numpy as np
        n = len(doubt_set)
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i, s in enumerate(doubt_set):
            # transitions from s
            out_t = [t for t in transitions if t[1] == s]
            if not out_t:
                # no transitions => row remains zero
                continue
            # We'll pick the first transition for simplicity
            trans_type, dep, act, dests, weights = out_t[0]
            total = sum(weights)
            for dest, w in zip(dests, weights):
                prob = w / total if total else 0
                if dest in doubt_set:
                    j = doubt_set.index(dest)
                    A[i, j] += prob
                elif dest in win_set:
                    b[i] += prob

        I_minus_A = np.eye(n) - A
        # Solve (I - A) * x = b
        x = np.linalg.solve(I_minus_A, b)
        return x