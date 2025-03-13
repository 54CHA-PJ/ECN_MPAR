import sys
from colorama import Fore, Style
from antlr4 import *
from gramListener import gramListener
from gramParser import gramParser
import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog

class gramPrintListener(gramListener):
    """

    - MODEL CREATION         : states, actions, transitions, rewards
    
    - MODEL VISUALIZATION    : check, describe, get_matrix_mix
    
    - PROBABILITY OF WINNING :
        - PROBABILITY - SYMBOLIC     : proba_symbolic_MC, proba_symbolic_MDP, proba_symbolic
        - PROBABILITY - ITERATIVE    : proba_iterative
        - STATISTICAL - QUANTITATIVE : simulate_one_path, proba_statistical_quantitative
        - STATISTICAL - QUALITATIVE  : proba_statistical_quanlitative
        
    - REINFORCEMENT LEARNING : TODO
    """
    
    """------------------------------------------------------------
                            MODEL CREATION
    ------------------------------------------------------------"""
    
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
        
    """------------------------------------------------------------
                        MODEL VISUALIZATION
    ------------------------------------------------------------"""

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
        

    def get_matrix_mix(self):
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
    
    def get_matrix_MC(self):
        """ 
        Pour uniquement un modèle MC, fournit :
         - La matrice de transition A (N lignes, N colonnes)
         - Une liste de descriptions des transitions 
        """
        N = len(self.states)
        A = np.zeros((N, N))
        desc = []
        for i, s in enumerate(self.states):
            desc.append(s)
            # On cherche une transition MC partant de s
            for (t_type, dep, act, dests, weights) in self.transitions:
                if t_type == "MC" and dep == s:
                    total = sum(weights)
                    probs = [w / total for w in weights] if total > 0 else [0] * len(weights)
                    for d, p in zip(dests, probs):
                        j = self.states.index(d)
                        A[i, j] = p
                    break  # on suppose qu'il n'y a qu'une seule transition MC par état
        return A, desc

    def get_matrix_MDP(self):
        """ 
        Pour un modèle MC ou MP
         - La matrice de transition A (N*A lignes, N colonnes)
         - Une liste de descriptions des transitions 
        """
        N = len(self.states)
        A_count = len(self.actions)
        M = np.zeros((N * A_count, N))
        desc = []
        for i, s in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                row_index = i * A_count + a_idx
                desc.append(f"[{s} | {action}]")
                # Chercher une transition MDP pour cet état et cette action
                mdp_transitions = [t for t in self.transitions if t[0] == "MDP" and t[1] == s and t[2] == action]
                if mdp_transitions:
                    t = mdp_transitions[0]
                    _, dep, act, dests, weights = t
                    total = sum(weights)
                    probs = [w / total for w in weights] if total > 0 else [0] * len(weights)
                    for d, p in zip(dests, probs):
                        j = self.states.index(d)
                        M[row_index, j] = p
                else:
                    # Sinon, si une transition MC existe pour s, on l'utilise
                    mc_transitions = [t for t in self.transitions if t[0] == "MC" and t[1] == s]
                    if mc_transitions:
                        t = mc_transitions[0]
                        _, dep, act, dests, weights = t
                        total = sum(weights)
                        probs = [w / total for w in weights] if total > 0 else [0] * len(weights)
                        for d, p in zip(dests, probs):
                            j = self.states.index(d)
                            M[row_index, j] = p
        return M, desc

        
    """------------------------------------------------------------
                      PROBABILITY OF WINNING
    ------------------------------------------------------------"""

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

    # ------------------------------------------------------------
    # PROBABILITY - SYMBOLIC APPROACH 
    # ------------------------------------------------------------

    def proba_symbolic_MC(self, win_set, lose_set, doubt_set):
        """
        MC: Solve (I-A)x=b to get win probabilities for uncertain states (doubt_set)
            - A: matrix of transition probabilities in doubt_set
            - b: vector of direct win probabilities from uncertain states
            - Using LINPROG
        """
        # If no uncertain states, it is already solved
        if not doubt_set:
            return []
        A, _ = self.get_matrix_MC()                         # Transition matrix
        doubt_list = list(doubt_set)                        # List of doubt states
        d_idx = {s: i for i, s in enumerate(doubt_list)}                        # Index of each doubt state
        w_index = [self.states.index(s) for s in win_set if s in self.states]   # Index of win states
        # Build equality constraints : 
        # (I-A) * x = b
        # x(s) - sum_{j in doubt} A[s,j]* x(j) = sum_{j in W} A[s,j]
        A_ub = []   # Coeffs of the Left side of the equation
        b_ub = []   # Right side of the equation
        for s in doubt_list:
            si = self.states.index(s)
            lhs = [0]*len(doubt_list)
            lhs[d_idx[s]] = 1
            for s2 in doubt_list:
                j = self.states.index(s2)
                lhs[d_idx[s2]] -= A[si, j]
            sumW = sum(A[si, wj] for wj in w_index)
            A_ub.append(lhs)
            b_ub.append(sumW)
            A_ub.append([-x for x in lhs])
            b_ub.append(-sumW)
        # Bounds of the variables (probabilities)
        bounds = [(0,1)]*len(doubt_list)
        # Objective : equality constraints (c=0)
        c = [0]*len(doubt_list)
        # Solve using LINPROG
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Symbolic approach failed.")
            return [0]*len(doubt_list)
        x_sol = res.x
        # Round to 0 or 1 if close enough
        for i in range(len(x_sol)):
            if x_sol[i] < 1e-8: x_sol[i] = 0
            elif x_sol[i] > 1-(1e-8): x_sol[i] = 1
        return [x_sol[d_idx[s]] for s in doubt_list]

    def proba_symbolic_MDP(self, win_set, lose_set, doubt_set):
        """
        Solve MDP maximum reachability probabilities via linprog.
            - Impose (I-A)x >= b for EVERY action of s.
            - Minimize sum x(s) for s in doubt_set.
        """
        # If no uncertain states, it is already solved
        if not doubt_set:
            return []
        # Get the transition matrix
        M, desc = self.get_matrix_MDP()                         # MDP transition matrix
        A_count = len(self.actions)                             # Number of actions
        doubt_list = list(doubt_set)                            # List of doubt states
        d_idx = {s: i for i, s in enumerate(doubt_list)}                        # Index of each doubt state
        w_index = [self.states.index(s) for s in win_set if s in self.states]   # Index of win states
        # Build >= constraints for each doubt state s, for each action
        # x(s) >= sum_{w in W} M[row, w] + sum_{d in doubt} M[row, d]* x(d)
        A_ub = []
        b_ub = []
        for s in doubt_list:
            s_index = self.states.index(s)
            row_base = s_index * A_count
            for a_idx in range(A_count):
                row = row_base + a_idx
                lhs = [0]*len(doubt_list)
                for d in doubt_list:
                    dcol = self.states.index(d)
                    lhs[d_idx[d]] += M[row, dcol]
                lhs[d_idx[s]] -= 1
                sumW = sum(M[row, wcol] for wcol in w_index)
                A_ub.append(lhs)
                b_ub.append(-sumW)
        # Bounds of the variables (probabilities)
        bounds = [(0, 1)] * len(doubt_list)
        # Objective : minimize sum x(s)
        c = [1.0]*len(doubt_list)                               
        # Solve using LINPROG
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Symbolic MDP approach failed: {res.message}")
            return [0.0]*len(doubt_list)
        x_sol = res.x
        # Round to 0 or 1 if close enough
        for i in range(len(x_sol)):
            if abs(x_sol[i]) < 1e-9:
                x_sol[i] = 0.0
            elif abs(x_sol[i] - 1.0) < 1e-9:
                x_sol[i] = 1.0
        return [x_sol[d_idx[s]] for s in doubt_list]

    def proba_symbolic(self, win_set, lose_set, doubt_set):
        # Check if there is any MDP transition or not
        has_mdp = any(t[0] == "MDP" for t in self.transitions)
        if not has_mdp:
            return self.proba_symbolic_MC(win_set, lose_set, doubt_set)
        else:
            return self.proba_symbolic_MDP(win_set, lose_set, doubt_set)
        
    # ------------------------------------------------------------
    # PROBABILITY - ITERATIVE APPROACH 
    # ------------------------------------------------------------

    def proba_iterative_MC(self, win_set, lose_set, doubt_set, tolerance=1e-8, max_iter=1000):
        """
        Iterative approach: 
         - Begin with known values (win=1, lose=0) and update uncertain states until convergence. 
            - For MC states, update via the weighted sum
            - For MDP states, update via the minimum over actions.
        """
        if not doubt_set:
            return []
        A, _ = self.get_matrix_MC()
        N = len(self.states)
        x = np.zeros(N)
        for i, s in enumerate(self.states):
            if s in win_set: x[i] = 1
            elif s in lose_set: x[i] = 0
        for _ in range(max_iter):
            diff = 0
            x_new = x.copy()
            for s in doubt_set:
                i = self.states.index(s)
                # x[i] = sum_{j} A[i,j]* x[j] (since W=1, L=0 are already in x)
                ssum = 0
                for j in range(N):
                    ssum += A[i,j]* x[j]
                x_new[i] = ssum
            diff = np.max(np.abs(x_new - x))
            x = x_new
            if diff < tolerance: break
        return [x[self.states.index(s)] for s in doubt_set]

    def proba_iterative_MDP(self, win_set, lose_set, doubt_set, tolerance=1e-8, max_iter=1000):
        """ Same process as proba_iterative_MC, but with MDP transitions 
            - Update uncertain states via the minimum over actions.
            - MDP states are updated via the minimum over actions.
            
        """
        if not doubt_set:
            return []
        M, _ = self.get_matrix_MDP()
        N = len(self.states)
        A_count = len(self.actions)
        x = np.zeros(N)
        for i,s in enumerate(self.states):
            if s in win_set: x[i] = 1
            elif s in lose_set: x[i] = 0
        for _ in range(max_iter):
            diff = 0
            x_new = x.copy()
            for s in doubt_set:
                i = self.states.index(s)
                best = 0
                base = i*A_count
                for a_idx in range(A_count):
                    row = base + a_idx
                    val = M[row,:].dot(x)
                    if val > best:
                        best = val
                x_new[i] = best
            diff = np.max(np.abs(x_new - x))
            x = x_new
            if diff < tolerance: break
        return [x[self.states.index(s)] for s in doubt_set]
    
    def proba_iterative(self, win_set, lose_set, doubt_set):
        # Check if there is any MDP transition or not
        has_mdp = any(t[0] == "MDP" for t in self.transitions)
        if not has_mdp:
            return self.proba_iterative_MC(win_set, lose_set, doubt_set)
        else:
            return self.proba_iterative_MDP(win_set, lose_set, doubt_set)
    
    # ------------------------------------------------------------
    # STATISTICAL - QUANTITATIVE
    # ------------------------------------------------------------
    
    def simulate_one_path(self, win_set, lose_set, initial_state="S0"):
        """ Simulate one path from initial_state to a win or lose state """
        current_state = initial_state
        while True:
            transitions = [t for t in self.transitions if t[1] == current_state]
            t_type = transitions[0][0]
            # Pour MC
            if t_type == "MC":
                # Normaliser les poids pour obtenir des probabilités
                weights = transitions[0][4]
                total = sum(weights)
                probabilities = [w / total for w in weights]
                # Choix aléatoire de l'état suivant
                next_state = np.random.choice(transitions[0][3], p=probabilities)
            # Pour MDP
            else:
                # Choix aléatoire de l'action
                actions = {t[2]: t for t in transitions}
                action = np.random.choice(list(actions.keys()))
                # Normaliser les poids pour obtenir des probabilités
                weights = actions[action][4]
                total = sum(weights)
                probabilities = [w / total for w in weights]
                # Choix aléatoire de l'état suivant
                next_state = np.random.choice(actions[action][3], p=probabilities)
            if next_state in win_set:
                return True
            if next_state in lose_set:
                return False
            current_state = next_state
    
    def proba_statistical_quantitative(self, win_set, lose_set, doubt_set, epsilon, delta):
        # Calcul du nombre de simulations requis à partir de la borne de Chernoff–Hoeffding
        N = int(np.ceil((np.log(2) - np.log(delta)) / ((2 * epsilon) ** 2)))
        print(f"[STATS] [ ε = {epsilon} | δ = {delta} ] -> {N} Simulations")
        # On récupère l'ensemble des états perdants (pour simulation) à partir de l'analyse
        _, lose_set, _ = self.get_state_analysis(win_set)
        prob_estimates = {}
        # On simule N fois pour chaque état incertain
        for state in doubt_set:
            wins = 0
            for _ in tqdm(range(N), desc=f"Simulations en cours ..."):
                if self.simulate_one_path(win_set, lose_set, initial_state=state):
                    wins += 1
            prob = wins / N
            prob_estimates[state] = prob
            print(f"État {state} : probabilité de gagner ≈ {prob:.4f}")
        
        return [prob_estimates[state] for state in doubt_set]
    
    # ------------------------------------------------------------
    # STATISTICAL - QUALITATIVE
    # ------------------------------------------------------------

    def proba_statistical_qualitative(self):
        return
    
    
    """------------------------------------------------------------
                        REINFORCEMENT LEARNING
    ------------------------------------------------------------"""