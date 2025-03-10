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
                G.add_edge(dep, dest, label=label, color=color, arrow_type=t_type)
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
    
    def draw_better_edges(self, G, pos):
        """  Draw edges with small arcs and labels with a white contour """
        groups = {}
        for u, v, d in G.edges(data=True):
            groups.setdefault((u, v), []).append(d)
        base_offset = 0.33
        opp = {(u, v) for (u, v) in groups if (v, u) in groups}

        for (u, v), eds in groups.items():
            if u == v:

                fixed_curvature = 0.3  
                for d in eds:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        edge_color=[d['color']],
                        connectionstyle=f'arc3, rad={fixed_curvature}',
                        arrows=True, arrowstyle='-|>', arrowsize=15,
                        ax=self.ax
                    )
                    x, y = pos[u]
                    self.ax.text(x, y + 0.3, "⟳ " + d['label'], fontsize=7,
                                 color=d['color'], ha='center', va='center',
                                 path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")])

            else:
                # --- Non-loop edges ---
                offs = [((i - (len(eds) - 1) / 2) * base_offset) for i in range(len(eds))]
                adjust = ((u, v) in opp) and (u > v)
                for off, d in zip(offs, eds):
                    cur = -off if d.get('arrow_type') == "MC" else off
                    if adjust:
                        cur += base_offset if cur >= 0 else -base_offset
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        edge_color=[d['color']],
                        arrows=True, arrowstyle='-|>', arrowsize=15,
                        connectionstyle=f'arc3, rad={cur}', ax=self.ax
                    )
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    # Place label closer to the source (using t = 0.3 along the edge)
                    t = 0.3
                    xm = x1 + t * (x2 - x1)
                    ym = y1 + t * (y2 - y1)
                    dx, dy = x2 - x1, y2 - y1
                    dist = np.sqrt(dx**2 + dy**2) or 1
                    perp = (dy/dist, -dx/dist)
                    self.ax.text(xm + cur * perp[0], ym + cur * perp[1],
                                 d['label'], fontsize=7, color=d['color'],
                                 ha='center', va='center',
                                 path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
    
    # ------------------------------
    # AUXILIARY FUNCTION : ANIMATION STEP
    
    def plot_simulation_state(self, model, current_state, chosen_edge=None):
        """ Redraws the model and highlights one edge """
        self.ax.clear()
        self.ax.axis('off')
        G = self.build_graph(model)
        states = model.states
        pos = self.get_positions(G, states)
        # Mark the current state in red; all others in lightgray.
        node_colors = ['red' if n == current_state else 'lightgray' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        self.draw_better_edges(G, pos)
        if chosen_edge is not None:
            (t_type, dep, act, dest) = chosen_edge
            if (dep, dest) in G.edges():
                groups = {}
                for u, v, d in G.edges(data=True):
                    groups.setdefault((u, v), []).append(d)
                if (dep, dest) in groups:
                    eds = groups[(dep, dest)]
                    base_offset = 0.33
                    offs = [((i - (len(eds) - 1) / 2) * base_offset) for i in range(len(eds))]
                    opp = (dest, dep) in groups
                    adjust = (opp and dep > dest)
                    index = 0
                    for i, d in enumerate(eds):
                        if d.get('arrow_type') == t_type:
                            index = i
                            break
                    off = offs[index]
                    cur = -off if t_type == "MC" else off
                    if adjust:
                        cur += base_offset if cur >= 0 else -base_offset
                    chosen_color = eds[index].get('color', 'red')
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[(dep, dest)],
                        edge_color=chosen_color,
                        width=3,
                        connectionstyle=f'arc3, rad={cur}',
                        ax=self.ax
                    )
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
                             color=color, fontweight=fontweight, fontsize=fontsize)
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
        win_states = [s.strip() for s in text.split(",") if s.strip()]
        # 2. Compute state analysis
        w, l, inc = self.model.get_state_analysis(win_states)
        print(Fore.LIGHTRED_EX + "\n-------------- Probability Calculation --------------" + Style.RESET_ALL)
        print(Fore.LIGHTRED_EX + f"Win States :  {Style.RESET_ALL}{w}")
        print(Fore.LIGHTRED_EX + f"Lose States : {Style.RESET_ALL}{l}")
        print(Fore.LIGHTRED_EX + f"Incertitude : {Style.RESET_ALL}{inc}")
        # 3. Let user choose which method
        method_options = ["Symbolic", "Iterative", "Statistical"]
        method, ok2 = QInputDialog.getItem(
            self, "Probability Method",
            "Which method ?",
            method_options, 0, False
        )
        if not ok2:
            return 
        if not inc:
            print("\n[PROBA] No incertitude states, no computation needed.")
            return
        # Main function call
        if method == "Symbolic":
            probs = self.model.proba_symbolic(w, inc)
        elif method == "Iterative":
            probs = self.model.proba_iterative(w, inc)
        else: 
            probs = self.model.proba_statistical(w, inc)

        print(Fore.LIGHTMAGENTA_EX + f"Method : {Fore.LIGHTYELLOW_EX}{method}" + Style.RESET_ALL)
        for st, val in zip(inc, probs):
            print(f"   - {Fore.LIGHTYELLOW_EX}{st}{Style.RESET_ALL} : {val:.4f}")
        print(Fore.LIGHTRED_EX + "-----------------------------------------------------" + Style.RESET_ALL)
        # Build full state probability dictionary: win=1, lose=0, inc from computed vector.
        state_prob = {s: 1.0 for s in w}
        state_prob.update({s: 0.0 for s in l})
        for s, p in zip(inc, probs):
            state_prob[s] = p
        self.model.state_prob = state_prob
        # Update the graph to show the new probabilities
        self.canvas.plot_model(self.model)
        
    def hide_probabilities(self):
        self.model.state_prob = {}
        self.canvas.plot_model(self.model)

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
        rows, desc, cols = self.model.get_matrix()
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
