import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QDoubleSpinBox, QLabel
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramParser import gramParser
from gramListener import gramListener
from colorama import Fore, Style, init as colorama_init

matplotlib.use('Qt5Agg')
DEFAULT_FILE = "ex.mdp"

# --------------------
# GRAMMAR PARSER

class gramPrintListener(gramListener):
    """ 
    Fields:
    - states: set of states (strings)
    - actions: set of actions (strings)
    - transitions:
        - type: "MDP" or "MC"
        - dep: departure state
        - act: action (if MDP)
        - dest_states: list of destination states
        - weights: list of weights
    """
    def __init__(self):
        super().__init__()
        self.states = set()
        self.actions = set()
        self.transitions = []

    def enterDefstates(self, ctx):
        self.states = {str(x) for x in ctx.ID()}

    def enterDefactions(self, ctx):
        self.actions = {str(x) for x in ctx.ID()}

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MDP", dep, act, ids, weights))

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.transitions.append(("MC", dep, None, ids, weights))
        
    def describe(self):
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "States: " + Style.RESET_ALL, self.states)
        print(Fore.LIGHTBLUE_EX + "Actions: " + Style.RESET_ALL, self.actions)
        print(Fore.LIGHTBLUE_EX + "Transitions:" + Style.RESET_ALL)
        for t in self.transitions:
            print(" -", t)
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------" + Style.RESET_ALL)
            
    def validate(self):
        valid = True
        s0_exists = False
        for t in self.transitions:
            if t[1] == "S0":
                s0_exists = True
            # Check if an action is not declared
            if t[2] and t[2] not in self.actions:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Action {Fore.YELLOW}{t[2]}{Style.RESET_ALL} not defined!")
            # Check if a state is not declared
            if t[1] not in self.states:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} State {Fore.YELLOW}{t[1]}{Style.RESET_ALL} not defined!")
            # Check if a destination state is not declared
            for s in t[3]:
                if s not in self.states:
                    print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} state {Fore.YELLOW}{s}{Style.RESET_ALL} not defined!")
            # Check if a weight is negative
            if any(z < 0 for z in t[4]):
                print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Negative weights detected in transition {Fore.LIGHTRED_EX}{t}{Style.RESET_ALL}")
                valid = False
            # Check if there is a state with both MDP and MC transitions
            if t[0] == "MC":
                for other in self.transitions:
                    if other[0] == "MDP" and t[1] == other[1]:
                        print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Cannot have MDP and MC transitions in the same state {Fore.LIGHTRED_EX}({t[1]}){Style.RESET_ALL}")
                        valid = False
        # Check if S0 is defined somewhere
        if not s0_exists:
            print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Initial state {Fore.LIGHTRED_EX}S0{Style.RESET_ALL} not defined! It is required for the simulation.")
            valid = False
        # Return the validity of the model
        return valid

    def check(self, name):
        """ Check model validity and print results
        Args:
            model (gramPrintListener): model to check
            name (str): file name
        """
        print(Fore.LIGHTBLUE_EX + f"\nModel: {name}" + Style.RESET_ALL)
        self.describe()
        if not self.validate():
            print(Fore.LIGHTRED_EX + "Model is not valid! Quitting..." + Style.RESET_ALL)
            sys.exit(1)
        print(Fore.LIGHTGREEN_EX + "Model is valid!" + Style.RESET_ALL)
        
    def get_matrix(self):
        """ Generate transition matrix
        - MC : Basic line with the probabilities
        - MDP : One line for each action, and the probabilities
        Returns:
            mat(numpy.ndarray): transition matrix
        """
        import numpy as np
        state_list = sorted(list(self.states))
        rows = [] # List of rows
        desc = [] # List of row info
        for idx, t in enumerate(self.transitions):
            trans_type, dep, act, dest_states, weights = t
            total = sum(weights)
            row = np.zeros(len(state_list))
            for dest, w in zip(dest_states, weights):
                prob = w / total if total != 0 else 0
                row[state_list.index(dest)] = prob
            if trans_type == "MDP":
                desc.append(f"[{idx}] ({dep}, {act})\t")
            else:
                desc.append(f"[{idx}] ({dep})\t")
            rows.append(row)
        return rows, desc, state_list
        
# --------------------
# USER INTERFACE

# Class for plotting the model
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.show_welcome()

    def show_welcome(self):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.text(
            0.5, 0.6, "Welcome to the MC/MDP Simulator!",
            ha='center', va='center', transform=self.ax.transAxes, fontsize=16
        )
        self.ax.text(
            0.5, 0.4, "made by Sacha Cruz and Jun Leduc",
            ha='center', va='center', transform=self.ax.transAxes, fontsize=12
        )
        self.draw()

    def plot_model(self, model):
        self.ax.clear()
        self.ax.axis('off')
        G = self.build_graph(model)
        pos = nx.spring_layout(G, seed=0)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        self.draw_better_edges(G, pos)
        self.draw()

    def build_graph(self, model):
        """ Build the model as a directed graph
        Each transition is added as an edge with attributes:
         - label: shows action (if MDP) and the probability
         - color: blue (MDP) or red (MC)
         - arrow_type: "MDP" or "MC"
        """
        G = nx.MultiDiGraph()
        for t in model.transitions:
            _, dep, action, dest_states, weights = t
            total = sum(weights)
            probs = [w / total for w in weights]
            arrow_type = "MDP" if action else "MC"
            for i, dest in enumerate(dest_states):
                label = f"[{action}]\n{probs[i]:.2f}" if action else f"{probs[i]:.2f}"
                color = 'blue' if action else 'red'
                G.add_edge(dep, dest, label=label, color=color, arrow_type=arrow_type)
        return G

    def draw_better_edges(self, G, pos):
        """ Draw edges with better curves and labels
        Args:
            G (nx.Graph): graph of the model
            pos (dict): positions of the nodes
        """
        groups = {}
        for u, v, d in G.edges(data=True):
            groups.setdefault((u, v), []).append(d)
        base_offset = 0.33
        opp = {(u, v) for (u, v) in groups if (v, u) in groups}
        for (u, v), eds in groups.items():
            offs = [((i - (len(eds) - 1) / 2) * base_offset) for i in range(len(eds))]
            adjust = ((u, v) in opp) and (u > v)
            for off, d in zip(offs, eds):
                cur = -off if d.get('arrow_type') == "MC" else off
                if adjust:
                    cur += base_offset if cur >= 0 else -base_offset
                if u == v:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        edge_color=[d['color']],
                        connectionstyle=f'arc3, rad={0.2 + abs(cur)}', ax=self.ax
                    )
                    x, y = pos[u]
                    self.ax.text(x + cur, y + 0.22, d['label'], fontsize=10,
                                 color=d['color'], ha='center', va='center')
                else:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        edge_color=[d['color']],
                        connectionstyle=f'arc3, rad={cur}', ax=self.ax
                    )
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1
                    L = (dx ** 2 + dy ** 2) ** 0.5 or 1
                    perp = (dy / L, -dx / L)
                    self.ax.text(xm + cur * 0.5 * perp[0], ym + cur * 0.5 * perp[1],
                                 d['label'], fontsize=10, color=d['color'],
                                 ha='center', va='center')

    def plot_simulation_state(self, model, current_state, chosen_edge=None):
        self.ax.clear()
        self.ax.axis('off')
        G = self.build_graph(model)
        pos = nx.spring_layout(G, seed=0)
        node_colors = ['red' if n == current_state else 'lightgray' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        self.draw_better_edges(G, pos)
        if chosen_edge is not None:
            chosen_type, chosen_dep, chosen_action, chosen_dest = chosen_edge
            groups = {}
            for u, v, d in G.edges(data=True):
                groups.setdefault((u, v), []).append(d)
            if (chosen_dep, chosen_dest) in groups:
                eds = groups[(chosen_dep, chosen_dest)]
                base_offset = 0.33
                offs = [((i - (len(eds) - 1) / 2) * base_offset) for i in range(len(eds))]
                opp = {(u, v) for (u, v) in groups if (v, u) in groups}
                adjust = ((chosen_dep, chosen_dest) in opp) and (chosen_dep > chosen_dest)
                index = 0
                for i, d in enumerate(eds):
                    if chosen_type == "MDP":
                        if d.get('label', '').startswith(f"[{chosen_action}]"):
                            index = i
                            break
                    else:
                        index = i
                        break
                off = offs[index]
                cur = -off if eds[index].get('arrow_type') == "MC" else off
                if adjust:
                    cur += base_offset if cur >= 0 else -base_offset
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(chosen_dep, chosen_dest)],
                    edge_color='red', width=3,
                    connectionstyle=f'arc3, rad={cur}', ax=self.ax
                )
        self.draw()

# --------------------
# MAIN INTERFACE

# Main interface of the application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MDP/MC Simulator")
        self.resize(800, 600)
        self.canvas = PlotCanvas(self, width=10, height=6, dpi=100)
        self.model = None  # Current loaded model
        self.simulation_running = False  # Simulation state
        self.current_state = "S0"  # Current (Initial) state in simulation
        colorama_init()
        
        # Control buttons
        
        # Load custom model
        self.loadButton = QPushButton("Load Model File")  
        self.loadButton.clicked.connect(self.load_file)
        # Load example model
        self.exampleButton = QPushButton("Use Example")  
        self.exampleButton.clicked.connect(self.load_example)
        # Start/stop simulation
        self.simulateButton = QPushButton("Launch Simulation!")  
        self.simulateButton.setStyleSheet("background-color: lightgreen")
        self.simulateButton.clicked.connect(self.toggle_simulation)
        # Print transition matrix
        self.printMatrixButton = QPushButton("Print Matrix")  
        self.printMatrixButton.clicked.connect(self.print_matrix)
        # Simulation delay controls
        self.delayLabel = QLabel("Transition Delay :")
        self.delaySpinBox = QDoubleSpinBox()
        self.delaySpinBox.setRange(0.05, 5.0)
        self.delaySpinBox.setSingleStep(0.05)
        self.delaySpinBox.setValue(0.5)
        
        # Layout setup
        
        # Horizontal layout for controls
        hlayout = QHBoxLayout()  
        hlayout.addWidget(self.loadButton)
        hlayout.addWidget(self.exampleButton)
        hlayout.addWidget(self.simulateButton)
        hlayout.addWidget(self.printMatrixButton)
        hlayout.addWidget(self.delayLabel)
        hlayout.addWidget(self.delaySpinBox)
        # Vertical layout for main window
        layout = QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.canvas)
        # Set central widget
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # Load model from MDP file
    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "MDP Files (*.mdp);;All Files (*)")
        if fname:
            self.process_file(fname)

    # Load example model (ex.mdp)
    def load_example(self):
        self.process_file(DEFAULT_FILE)

    # Convert file content to model 
    def process_file(self, fname):
        # Open file
        with open(fname, 'r') as f:
            content = f.read()
        # Parse content
        stream = InputStream(content)
        lexer = gramLexer(stream)
        tokens = CommonTokenStream(lexer)
        parser = gramParser(tokens)
        tree = parser.program()
        # Build model
        model = gramPrintListener()
        walker = ParseTreeWalker()
        walker.walk(model, tree)
        # Check model validity
        model.check(fname)
        # Plot model graph in canvas
        self.model = model
        self.current_state = "S0"
        self.canvas.plot_model(model)

    # --------------------
    # SIMULATION FUNCTIONS
    
    def toggle_simulation(self):
        if not self.simulation_running:
            self.start_simulation()
        else:
            self.stop_simulation()

    def start_simulation(self):
        if self.model is None:
            print(Fore.LIGHTRED_EX + "No model loaded!" + Style.RESET_ALL)
            return
        self.simulation_running = True
        self.current_state = "S0"
        self.simulateButton.setText("Stop Simulation")
        self.simulateButton.setStyleSheet("background-color: red")
        print(f"\n[SIM] Simulation begins in state {self.current_state}")
        QTimer.singleShot(0, self.simulate_step)

    def stop_simulation(self):
        self.simulation_running = False
        self.simulateButton.setText("Start simulation")
        self.simulateButton.setStyleSheet("background-color: lightgreen")
        self.canvas.plot_model(self.model)
        print("[SIM] Simulation stopped.")

    def simulate_step(self):
        if not self.simulation_running:
            return
        outgoing = []
        # Collect all outgoing transitions from the current state
        for t in self.model.transitions:
            if t[1] == self.current_state:
                for dest in t[3]:
                    outgoing.append((t[0], t[1], t[2], dest))

        # STOP if there are no outgoing transitions
        if not outgoing:
            print(Fore.LIGHTYELLOW_EX + f"[SIM] State {self.current_state} has no transitions!" + Style.RESET_ALL)
            self.stop_simulation()
            return

        # STOP if the state is in a loop
        if len(outgoing) == 1 and outgoing[0][1] == outgoing[0][3]:
            print(Fore.LIGHTYELLOW_EX + f"[SIM] State {self.current_state} is in a loop!" + Style.RESET_ALL)
            self.stop_simulation()
            return

        # Otherwise, randomly choose an outgoing transition
        chosen_edge = random.choice(outgoing)
        if chosen_edge[0] == "MDP":
            print(f"[MDP] [{chosen_edge[2]}] {chosen_edge[1]} -> {chosen_edge[3]}")
        else:
            print(f"[MC] {chosen_edge[1]} -> {chosen_edge[3]}")
        self.current_state = chosen_edge[3]
        self.canvas.plot_simulation_state(self.model, self.current_state, chosen_edge)
        delay = self.delaySpinBox.value()
        QTimer.singleShot(int(delay * 1000), self.simulate_step)

    def print_matrix(self):
        if self.model is None:
            print(Fore.LIGHTRED_EX + "No model loaded!" + Style.RESET_ALL)
            return
        rows, desc, cols = self.model.get_matrix()
        
        print(Fore.LIGHTBLUE_EX + "\nTransition Matrix:" + Style.RESET_ALL)
        print("Columns:\t", cols)
        for ind, row in enumerate(rows):
            print(desc[ind], row)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    if len(sys.argv) >= 2:
        fname = sys.argv[1]
        win.process_file(fname)
    win.show()
    sys.exit(app.exec_())
