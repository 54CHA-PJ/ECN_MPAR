import sys
from colorama import Fore, Style
from antlr4 import *
from gramListener import gramListener
from gramParser import gramParser
import numpy as np

class gramPrintListener(gramListener):
    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
        self.transitions = []
        self.rewards = {}
        self.warning_state_list = []

    def enterDefstates(self, ctx: gramParser.DefstatesContext):
        # Process state definitions and assign rewards (default 0 if not provided)
        for sctx in ctx.statedef():
            state_name = sctx.ID().getText()
            rew = int(sctx.INT().getText()) if sctx.INT() else 0
            if not sctx.INT():
                self.warning_state_list.append(state_name)
            self.states.append(state_name)
            self.rewards[state_name] = rew

    def enterDefactions(self, ctx: gramParser.DefactionsContext):
        # Store declared actions
        self.actions = [str(x) for x in ctx.ID()]

    def enterTransact(self, ctx: gramParser.TransactContext):
        # Process MDP transitions (with an action)
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MDP", dep, act, ids, weights))

    def enterTransnoact(self, ctx: gramParser.TransnoactContext):
        # Process MC transitions (without action)
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MC", dep, None, ids, weights))

    def validate(self):
        valid = True
        # Check for initial state S0
        if "S0" not in self.states:
            print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} No initial state 'S0' found.")
            valid = False
        # Validate transitions, actions, and weights
        for t_type, dep, act, dests, weights in self.transitions:
            if act and act not in self.actions:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Action {Fore.YELLOW}{act}{Style.RESET_ALL} not declared.")
                self.actions.append(act)
            for d in dests:
                if d not in self.states:
                    print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Destination state {Fore.YELLOW}{d}{Style.RESET_ALL} unknown.")
                    self.states.append(d)
            if any(w < 0 for w in weights):
                print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Negative weight found in transition {dep} -> {dests}.")
                valid = False
            if t_type == "MC":
                for other in self.transitions:
                    if other[0] == "MDP" and other[1] == dep:
                        print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Mixed MC+MDP on state {dep}.")
                        valid = False
        for warn_state in self.warning_state_list:
            print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} State {Fore.YELLOW}{warn_state}{Style.RESET_ALL} has no reward (default 0).")
            self.rewards[warn_state] = 0
        return valid

    def check(self, filename):
        print(Fore.LIGHTBLUE_EX + f"\nModel: {filename}" + Style.RESET_ALL)
        if not self.validate():
            print(Fore.LIGHTRED_EX + "Model is not valid! Quitting..." + Style.RESET_ALL)
            sys.exit(1)
        print(Fore.LIGHTGREEN_EX + "Model is valid." + Style.RESET_ALL)
        self.describe()

    def describe(self):
        # Brief model description: states, rewards, actions, transitions
        print(Fore.LIGHTBLUE_EX + "\n------------------ Model Description ----------------" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "States  :" + Style.RESET_ALL, self.states)
        print(Fore.LIGHTBLUE_EX + "Rewards :" + Style.RESET_ALL, self.rewards)
        print(Fore.LIGHTBLUE_EX + "Actions :" + Style.RESET_ALL, self.actions)
        print(Fore.LIGHTBLUE_EX + "Transitions :" + Style.RESET_ALL)
        for t in self.transitions:
            print("   ", t)
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)

    def get_matrix(self):
        # Build transition matrix for printing in the UI
        state_list = self.states[:]
        rows = []
        desc = []
        for idx, (t_type, dep, act, dests, weights) in enumerate(self.transitions):
            total = sum(weights)
            row = np.zeros(len(state_list))
            # Compute probability for each destination
            for d, w in zip(dests, weights):
                prob = w / total if total else 0
                row[state_list.index(d)] = prob
            label = f"[{idx}] ({dep}, {act})" if t_type == "MDP" else f"[{idx}]    ({dep})"
            desc.append(label)
            rows.append(row)
        return rows, desc, state_list

    def get_state_analysis(self, initial_win_states=["S0"]):
        # Classify states as winning, losing, or uncertain
        win_states = initial_win_states[:]
        changed = True
        while changed:
            changed = False
            # Process each state not already in win_states
            for s in self.states:
                if s in win_states:
                    continue
                out_t = [t for t in self.transitions if t[1] == s]
                if not out_t:
                    continue
                # MC transitions: all destinations must be winning
                mc_ok = all(all(d in win_states for d in t[3]) for t in out_t if t[0] == "MC")
                if not mc_ok:
                    continue
                # For MDP transitions: each action must only lead to winning states
                mdp_t = [t for t in out_t if t[0] == "MDP"]
                if mdp_t:
                    actions = set(tt[2] for tt in mdp_t)
                    mdp_ok = True
                    for a in actions:
                        same_action = [tt for tt in mdp_t if tt[2] == a]
                        for trans in same_action:
                            if any(dest not in win_states for dest in trans[3]):
                                mdp_ok = False
                                break
                        if not mdp_ok:
                            break
                    if not mdp_ok:
                        continue
                win_states.append(s)
                changed = True
        # Identify losing states: no transitions or only self-loops
        lose_states = []
        for s in self.states:
            if s in win_states:
                continue
            out_t = [t for t in self.transitions if t[1] == s]
            if not out_t or all(t[3] == [s] for t in out_t):
                lose_states.append(s)
        incertitude = [s for s in self.states if s not in win_states and s not in lose_states]
        return win_states, lose_states, incertitude
    
# --------------------------------------
# Probability calculation 
    
# SYMBOLIC APPROACH

    def proba_symbolic_MC(self, win_set, doubt_set):
        """
        MC: Solve (I-A)x=b to get win probabilities for uncertain states.
        A: matrix of transition probabilities among uncertain states.
        b: vector of probabilities to win states.
        """
        n = len(doubt_set)
        if n == 0:
            return []
        A = np.zeros((n, n))
        b = np.zeros(n)
        # Loop over each uncertain state
        for i, s in enumerate(doubt_set):
            # Get MC transitions from state s
            mc_trans = [t for t in self.transitions if t[0] == "MC" and t[1] == s]
            if not mc_trans:
                continue
            trans_type, dep, act, dests, weights = mc_trans[0]
            total = sum(weights)
            if total == 0:
                continue
            # Distribute probabilities to uncertain or win states
            for d, w in zip(dests, weights):
                prob = w / total
                if d in doubt_set:
                    j = doubt_set.index(d)
                    A[i, j] += prob
                elif d in win_set:
                    b[i] += prob
        # Print matrix for MC symbolic calculation
        print(Fore.LIGHTBLUE_EX + "\n--- Symbolic MC Transition Matrix ---" + Style.RESET_ALL)
        print("Matrix A:")
        print(A)
        print("Vector b:")
        print(b)
        # Solve linear system (I-A)x = b
        I_minus_A = np.eye(n) - A
        x = np.linalg.solve(I_minus_A, b)
        print("Solution x:")
        for s, prob in zip(doubt_set, x):
            print(f"{s}: {prob:.4f}")
        return x

    def proba_symbolic_MDP(self, win_set, lose_set, doubt_set):
        """
        MDP (symbolic approach): For each uncertain state, choose the action (row)
        that gives the lowest direct win probability. Then build a matrix A and vector b
        where, for state s, b(s) is the chosen action's win probability (i.e. sum of probabilities
        leading directly to win states) and A(s, t) holds the transition probabilities to uncertain states.
        Solve (I-A)x = b.
        """
        n = len(doubt_set)
        if n == 0:
            return []
        A = np.zeros((n, n))
        b = np.zeros(n)
        # Loop over each uncertain state s
        for i, s in enumerate(doubt_set):
            # Gather all transitions (actions) from state s
            actions = [t for t in self.transitions if t[1] == s]
            best_r = float('inf')
            best_row = np.zeros(n)
            # Evaluate each action row
            for (t_type, dep, act, dests, weights) in actions:
                total = sum(weights)
                if total == 0:
                    continue
                r = 0.0  # direct win probability for this action
                row = np.zeros(n)  # transition probabilities to uncertain states
                for d, w in zip(dests, weights):
                    prob = w / total
                    if d in win_set:
                        r += prob
                    elif d in doubt_set:
                        j = doubt_set.index(d)
                        row[j] += prob
                    # If destination is losing, contributes 0.
                # Choose the action with the smallest r (worst-case)
                if r < best_r:
                    best_r = r
                    best_row = row
            A[i, :] = best_row
            b[i] = best_r
        # Print the matrix for MDP symbolic calculation
        print(Fore.LIGHTBLUE_EX + "\n--- Symbolic MDP Transition Matrix (Selected Actions) ---" + Style.RESET_ALL)
        print("Matrix A:")
        print(A)
        print("Vector b:")
        print(b)
        I_minus_A = np.eye(n) - A
        x = np.linalg.solve(I_minus_A, b)
        print("Solution x:")
        for s, prob in zip(doubt_set, x):
            print(f"{s}: {prob:.4f}")
        return x

    def proba_symbolic(self, win_set, incertitude):
        # Dispatcher: choose MC or MDP method based on transitions present
        has_mdp = any(t[0] == "MDP" for t in self.transitions)
        if not has_mdp:
            return self.proba_symbolic_MC(win_set, incertitude)
        else:
            w, l, d = self.get_state_analysis(win_set)
            return self.proba_symbolic_MDP(w, l, d)
        
# ITERATIVE APPROACH

    def proba_iterative(self, win_set, doubt_set):
        print("[Iterative] Not implemented yet. Returning dummy zeros.")
        return np.zeros(len(doubt_set))
    
# STATISTICAL APPROACH

    def proba_statistical(self, win_set, doubt_set, trials=10000):
        print("[Statistical] Not implemented yet. Returning dummy zeros.")
        return np.zeros(len(doubt_set))
