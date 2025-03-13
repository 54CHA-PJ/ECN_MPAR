# main.py
import sys
import random
import networkx as nx
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
import numpy as np

# UI
from colorama import Fore, Style, init as colorama_init
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QDoubleSpinBox, QLabel, QInputDialog,
    QDialog, QDialogButtonBox, QVBoxLayout, QComboBox
)
matplotlib.use('Qt5Agg')

# Local imports
from gramLexer import gramLexer
from gramParser import gramParser
from mc_mdp import gramPrintListener

# Example file
DEFAULT_FILE = "./mdp/ex.mdp"
# Location of MDP files
MDP_DIR = "./mdp"

# -----------------------------------------------------------------
# GRAPH PLOTTING
# -----------------------------------------------------------------

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.show_welcome()

    def show_welcome(self):
        """ Show "Welcome" Message """
        self.ax.clear()
        self.ax.axis('off')
        self.ax.text(0.5, 0.6, "Welcome to the MC/MDP Simulator!",
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=16,
                     path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
        self.ax.text(0.5, 0.4, "Made by Sacha Cruz and Jun Leduc",
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=12,
                     path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
        self.draw()
        
    def plot_model(self, model):
        """
        Create and plots the model :
        - Build the graph       (build_graph)
        - Compute positions     (get_positions)
        - Normalize positions   (norm_positions)
        - Draw nodes and labels (nx.draw_networkx_*)
        - Draw edges            (draw_better_edges)
        - Show probabilities    (plot_proba)
        - Draw the graph        (draw)
        """
        self.ax.clear()
        self.ax.axis('off')
        G = self.build_graph(model)
        pos = self.get_positions(G, model.states) # Having Model States as a parameter allows to modify the order of appearance
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        self.draw_better_edges(G, pos)
        self.plot_proba(model, pos)
        self.draw()
    
    # ------------------------------
    # AUXILIARY FUNCTIONS : STATES
    # ------------------------------
    
    def build_graph(self, model):
        """ Build the model as a directed graph """
        G = nx.MultiDiGraph()
        for (t_type, dep, act, dests, weights) in model.transitions:
            total = sum(weights)
            probs = [w / total if total else 0 for w in weights]
            color = 'blue' if t_type == "MDP" else 'red'
            for p, dest in zip(probs, dests):
                label = f"[{act}]\n{p:.2f}" if act else f"{p:.2f}"
                # Store the action separately in the edge data so we can distinguish edges with the same (dep, dest) but different actions
                G.add_edge(
                    dep,
                    dest,
                    label=label,
                    color=color,
                    arrow_type=t_type,
                    action=act
                )
        return G

    def get_positions(self, G, states, layer_distance=3, root_name="S0"):
        """ Compute positions using a hierarchical layout, centered on S0 (by default) """
        # Define the root node
        if root_name in states:
            root = root_name
        elif states:
            root = states[0]
        else:
            return {}
        pos = {}
        # Compute shortest-path lengths from root.
        lengths = nx.single_source_shortest_path_length(G, root)
        # Group nodes by distance (layer)
        layers = {}
        for node, dist in lengths.items():
            layers.setdefault(dist, []).append(node)
        # Add unreachable nodes to an extra layer.
        unreachable = [node for node in G.nodes() if node not in lengths]
        if unreachable:
            layers.setdefault(max(layers.keys()) + 1, []).extend(unreachable)
        sorted_layers = sorted(layers.keys())
        # Place the root at the center.
        pos[root] = (0, 0)
        # Place the other nodes in layers around the root.
        for layer in sorted_layers:
            if layer == 0:
                continue
            nodes = layers[layer]
            n = len(nodes)
            angle_gap = 2 * np.pi / n if n > 0 else 0
            radius = layer * layer_distance
            for i, node in enumerate(nodes):
                angle = i * angle_gap
                pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        return pos

    # ------------------------------
    # AUXILIARY FUNCTIONS : EDGES
    # ------------------------------
    
    def draw_grouped_edges(self, G, pos, u, v, eds, highlight_edge=None, highlight_width=3):
        """
        Draw a group of edges from u->v (or a loop on u), possibly highlighting
        one specific edge. We now also check the 'action' so that only the exact
        chosen edge is highlighted (if multiple edges exist between the same pair).
        """
        base_offset = 0.33
        fixed_curvature = 0.3

        # If it's a loop edge
        if u == v:
            for d in eds:
                # Determine if this edge is the chosen one
                chosen_edge = (
                    highlight_edge
                    and highlight_edge[0] == d.get('arrow_type')  # t_type
                    and highlight_edge[1] == u                   # from-state
                    and highlight_edge[2] == d.get('action')     # action
                    and highlight_edge[3] == v                   # to-state
                )

                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)],
                    edge_color=[d['color']],
                    connectionstyle=f'arc3, rad={fixed_curvature}',
                    arrows=True, arrowstyle='-|>', arrowsize=15,
                    ax=self.ax,
                    width=(highlight_width if chosen_edge else 1)
                )
                x, y = pos[u]
                self.ax.text(x, y + 0.3, "⟳ " + d['label'], fontsize=7,
                             color=d['color'], ha='center', va='center',
                             path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")])
        else:
            # Non-loop edges
            offs = [((i - (len(eds) - 1) / 2) * base_offset) for i in range(len(eds))]
            opp = ((v, u) in G.edges())  # Check if there's an opposing edge v->u
            for i, d in enumerate(eds):
                off = offs[i]
                if d.get('arrow_type') == "MC":
                    off = -off
                if opp and u > v:
                    off += base_offset if off >= 0 else -base_offset

                # Now check action in addition to arrow_type, from-state, to-state
                chosen_edge = (
                    highlight_edge
                    and highlight_edge[0] == d.get('arrow_type')  # t_type
                    and highlight_edge[1] == u                   # from-state
                    and highlight_edge[2] == d.get('action')     # action
                    and highlight_edge[3] == v                   # to-state
                )

                color = d['color']
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)],
                    edge_color=[color],
                    arrows=True, arrowstyle='-|>', arrowsize=15,
                    connectionstyle=f'arc3, rad={off}',
                    ax=self.ax,
                    width=(highlight_width if chosen_edge else 1)
                )
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                t = 0.3
                xm = x1 + t * (x2 - x1)
                ym = y1 + t * (y2 - y1)
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx**2 + dy**2) or 1
                perp = (dy/dist, -dx/dist)
                self.ax.text(xm + off * perp[0], ym + off * perp[1],
                             d['label'], fontsize=7, color=color,
                             ha='center', va='center',
                             path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])

    def draw_better_edges(self, G, pos, highlight_edge=None):
        """ Draw edges with small arcs, labels with a white contour, and optional highlighting """
        # Group edges from u->v
        groups = {}
        for u, v, d in G.edges(data=True):
            groups.setdefault((u, v), []).append(d)

        # Draw each group with our unified function
        for (u, v), eds in groups.items():
            self.draw_grouped_edges(G, pos, u, v, eds, highlight_edge=highlight_edge)

    # ------------------------------
    # AUXILIARY FUNCTION : ANIMATION STEP
    # ------------------------------

    def plot_simulation_state(self, model, current_state, chosen_edge=None):
        """ Redraws the model and highlights one edge if chosen """
        self.ax.clear()
        self.ax.axis('off')
        G = self.build_graph(model)
        states = model.states
        pos = self.get_positions(G, states)
        # Mark the current state in red; all others in lightgray.
        node_colors = ['red' if n == current_state else 'lightgray' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        # Now we just call draw_better_edges once, optionally highlighting chosen_edge
        self.draw_better_edges(G, pos, highlight_edge=chosen_edge)
        self.draw()

        
    def plot_proba(self, model, pos):
        """ Show the probability of winning in each state """
        # Draw additional text: winning probability beside each state if available
        if hasattr(model, "state_prob"):
            for node, p in model.state_prob.items():
                x, y = pos.get(node, (0, 0))
                fontweight = "bold"
                if p == 0: 
                    text_str = "0"  
                    color = "red"
                else:
                    text_str = f"{p:.3f}"
                    color = "green" if node == "S0" else "black"
                    
                fontsize = 12 if node == "S0" else 8
                self.ax.text(x - 0.5, y - 0.5, text_str, 
                             color=color, fontweight=fontweight, fontsize=fontsize,
                             path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
# -----------------------------------------------------------------
# MAIN WINDOW
# -----------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MDP/MC Simulator")
        self.resize(800, 600)
        self.canvas = PlotCanvas(self, width=10, height=6, dpi=100)
        self.model = None
        self.simulation_running = False
        self.current_state = "S0"
        colorama_init()

        # Load MDP file
        self.loadButton = QPushButton("Load Model File")
        self.loadButton.clicked.connect(self.load_file)
        
        # Load example file
        self.exampleButton = QPushButton("Use Example")
        self.exampleButton.clicked.connect(self.load_example)

        # Print matrix 
        self.printMatrixButton = QPushButton("Print Matrix")
        self.printMatrixButton.clicked.connect(self.print_matrix)

        # Probability analysis 
        self.probabilityButton = QPushButton("Probability analysis")
        self.probabilityButton.clicked.connect(self.probability_analysis)
        self.hideProbabilityButton = QPushButton("Hide Probabilities")
        self.hideProbabilityButton.clicked.connect(self.hide_probabilities)
        self.hideProbabilityButton.hide()  # Hide the button by default
    
        # Simulation
        self.simulateButton = QPushButton("Launch Simulation!")
        self.simulateButton.setStyleSheet("background-color: lightgreen")
        self.simulateButton.clicked.connect(self.toggle_simulation)
        
        # Input for simulation speed
        self.delayLabel = QLabel("Transition Delay :")
        self.delaySpinBox = QDoubleSpinBox()
        self.delaySpinBox.setRange(0.05, 5.0)
        self.delaySpinBox.setSingleStep(0.05)
        self.delaySpinBox.setValue(0.5)

        # Layout setup
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.loadButton)
        hlayout.addWidget(self.exampleButton)
        hlayout.addWidget(self.printMatrixButton)
        hlayout.addWidget(self.probabilityButton)
        hlayout.addWidget(self.hideProbabilityButton)
        hlayout.addWidget(self.delayLabel)
        hlayout.addWidget(self.delaySpinBox)
        hlayout.addWidget(self.simulateButton)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.canvas)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # --------------------
    # File loading

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", MDP_DIR, "MDP Files (*.mdp);;All Files (*)"
        )
        if fname:
            self.process_file(fname)

    def load_example(self):
        self.process_file(DEFAULT_FILE)

    def process_file(self, fname):
        with open(fname, 'r') as f:
            content = f.read()
        stream = InputStream(content)
        lexer = gramLexer(stream)
        tokens = CommonTokenStream(lexer)
        parser = gramParser(tokens)
        tree = parser.program()
        model = gramPrintListener()
        walker = ParseTreeWalker()
        walker.walk(model, tree)
        model.check(fname)
        self.model = model
        self.current_state = "S0"
        self.canvas.plot_model(model)

    # --------------------
    # Probability analysis
    
    def probability_analysis(self):
        # If no model is loaded
        if not self.model:
            print(Fore.LIGHTRED_EX + "No model loaded!" + Style.RESET_ALL)
            return
        # 1. Prompt the user for win states
        text, ok = QInputDialog.getText( self, "Win States", "Enter win state(s), separate using \',\' :" )
        if not ok or not text.strip():
            return 
        win_states_initial = [s.strip() for s in text.split(",") if s.strip()]
        # 2. Compute state analysis
        win_set, lose_set, doubt_set = self.model.get_state_analysis(win_states_initial)
        print(Fore.LIGHTRED_EX + "\n-------------- Probability Calculation --------------" + Style.RESET_ALL)
        print(Fore.LIGHTRED_EX + f"Win States :  {Style.RESET_ALL}{win_set}")
        print(Fore.LIGHTRED_EX + f"Lose States : {Style.RESET_ALL}{lose_set}")
        print(Fore.LIGHTRED_EX + f"Incertitude : {Style.RESET_ALL}{doubt_set}")
        # 3. Let user choose which method
        method_options = ["Probability - Symbolic", "Probability - Iterative", "Statistical - Quantitative", "Statistical - Qualitative"]
        method, ok2 = QInputDialog.getItem(
            self, "Probability Calculation Method",
            "Which method ?",
            method_options, 0, False
        )
        if not ok2:
            return 
        if not doubt_set:
            print("\n[PROBA] No incertitude states, no computation needed.")
            print(Fore.LIGHTRED_EX + "-----------------------------------------------------" + Style.RESET_ALL)
            return
        print(Fore.LIGHTMAGENTA_EX + f"Method : {Fore.LIGHTYELLOW_EX}{method}\n" + Style.RESET_ALL)
        # 4. Main function call
        if method == "Probability - Symbolic":
            probs = self.model.proba_symbolic(win_set, lose_set, doubt_set)
        elif method == "Probability - Iterative":
            probs = self.model.proba_iterative(win_set, lose_set, doubt_set)
        elif method == "Statistical - Quantitative":
            dialog = QDialog(self)
            dialog.setWindowTitle("SMC Quantitative Parameters")
            layout = QVBoxLayout(dialog)
            epsilonLabel = QLabel("Enter precision (epsilon):")
            epsilonSpin = QDoubleSpinBox()
            epsilonSpin.setRange(0.0, 1.0)
            epsilonSpin.setDecimals(4)
            epsilonSpin.setValue(0.01)
            deltaLabel = QLabel("Enter error rate (delta):")
            deltaSpin = QDoubleSpinBox()
            deltaSpin.setRange(0.0, 1.0)
            deltaSpin.setDecimals(4)
            deltaSpin.setValue(0.01)
            layout.addWidget(epsilonLabel)
            layout.addWidget(epsilonSpin)
            layout.addWidget(deltaLabel)
            layout.addWidget(deltaSpin)
            buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(buttonBox)
            buttonBox.accepted.connect(dialog.accept)
            buttonBox.rejected.connect(dialog.reject)
            if dialog.exec_() != QDialog.Accepted:
                return
            epsilon = epsilonSpin.value()
            delta = deltaSpin.value()
            probs = self.model.proba_statistical_quantitative(win_set, lose_set, doubt_set, epsilon, delta)
        else:
            print(Fore.LIGHTRED_EX + "Method not implemented yet." + Style.RESET_ALL)
            return
        # 5. Show results
        print(Fore.LIGHTMAGENTA_EX + f"RESULTS : {Fore.LIGHTYELLOW_EX}{method}" + Style.RESET_ALL)
        for st, val in zip(doubt_set, probs):
            print(f"   - {Fore.LIGHTYELLOW_EX}{st}{Style.RESET_ALL} : {val:.4f}")
        print(Fore.LIGHTRED_EX + "-----------------------------------------------------" + Style.RESET_ALL)
        state_prob = {s: 1 for s in win_set}
        state_prob.update({s: 0 for s in lose_set})
        for s, p in zip(doubt_set, probs):
            state_prob[s] = p
        self.model.state_prob = state_prob
        # Update the graph to show the new probabilities
        self.canvas.plot_model(self.model)
        # Show the button to hide the probabilities
        self.hideProbabilityButton.show() 
        
    def hide_probabilities(self):
        self.model.state_prob = {}
        self.canvas.plot_model(self.model)
        self.hideProbabilityButton.hide() 

    # --------------------
    # Simulation

    def toggle_simulation(self):
        if not self.simulation_running:
            self.start_simulation()
        else:
            self.stop_simulation()

    def start_simulation(self):
        if not self.model:
            print(Fore.LIGHTRED_EX + "No model loaded!" + Style.RESET_ALL)
            return
        self.simulation_running = True
        self.current_state = "S0"
        self.simulateButton.setText("Stop Simulation")
        self.simulateButton.setStyleSheet("background-color: red")
        print(f"\n[SIM] Starting in state {self.current_state}")
        QTimer.singleShot(0, self.simulate_step)

    def stop_simulation(self):
        self.simulation_running = False
        self.simulateButton.setText("Start simulation")
        self.simulateButton.setStyleSheet("background-color: lightgreen")
        self.canvas.plot_model(self.model)
        print("[SIM] Simulation stopped.")

    def simulate_step(self):
        # STOP if the simulation is stopped
        if not self.simulation_running:
            return
        # Look at the outgoing transitions
        outgoing = []
        for t in self.model.transitions:
            if t[1] == self.current_state:
                for dest in t[3]:
                    outgoing.append((t[0], t[1], t[2], dest))
        # STOP if it's stucked
        if not outgoing:
            print(f"[SIM] No transitions from {self.current_state}, stopping.")
            self.stop_simulation()
            return
        # STOP if it's a loop
        if len(outgoing) == 1 and outgoing[0][3] == self.current_state:
            print(f"[SIM] {self.current_state} is in a loop => stopping.")
            self.stop_simulation()
            return
        # Choose a random transition
        chosen_edge = random.choice(outgoing)
        t_type, dep, act, dest = chosen_edge
        if t_type == "MDP":
            print(f"[MDP] {dep} -{act}-> {dest}")
        else:
            print(f"[MC]  {dep} ---> {dest}")
        self.current_state = dest
        self.canvas.plot_simulation_state(self.model, self.current_state, chosen_edge)
        delay = self.delaySpinBox.value()
        QTimer.singleShot(int(delay * 1000), self.simulate_step)

    def print_matrix(self):
        if not self.model:
            print(Fore.LIGHTRED_EX + "No model loaded!" + Style.RESET_ALL)
            return
        rows, desc, cols = self.model.get_matrix_mix()
        print(Fore.LIGHTBLUE_EX + "\n----------------- Transition Matrix -----------------" + Style.RESET_ALL)
        print("Columns :", cols)
        for d, row in zip(desc, rows):
            print(d, row)
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)

# --------------------
# Main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        win.process_file(fname)
    win.show()
    sys.exit(app.exec_())
