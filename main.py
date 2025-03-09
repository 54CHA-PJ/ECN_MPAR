# main.py
import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QDoubleSpinBox, QLabel, QInputDialog,
    QDialog, QDialogButtonBox, QVBoxLayout, QComboBox
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from colorama import Fore, Style, init as colorama_init
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker

# Grammar files
from gramLexer import gramLexer
from gramParser import gramParser

# Our code from mc_mdp.py
from mc_mdp import gramPrintListener

matplotlib.use('Qt5Agg')
DEFAULT_FILE = "ex.mdp"


class PlotCanvas(FigureCanvas):
    """For visualizing the model graph."""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.show_welcome()

    def show_welcome(self):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.text(0.5, 0.6, "Welcome to the MC/MDP Simulator!",
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=16)
        self.ax.text(0.5, 0.4, "made by Sacha Cruz and Jun Leduc",
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
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
        """ Build the model as a directed graph. """
        G = nx.MultiDiGraph()
        for (t_type, dep, act, dests, weights) in model.transitions:
            total = sum(weights)
            probs = [w / total if total else 0 for w in weights]
            color = 'blue' if t_type == "MDP" else 'red'
            for p, dest in zip(probs, dests):
                label = f"[{act}]\n{p:.2f}" if act else f"{p:.2f}"
                G.add_edge(dep, dest, label=label, color=color, arrow_type=t_type)
        return G

    def draw_better_edges(self, G, pos):
        """ Draw edges with small arcs + labels. """
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
                    dist = (dx**2 + dy**2)**0.5 or 1
                    perp = (dy/dist, -dx/dist)
                    self.ax.text(xm + cur*0.5*perp[0], ym + cur*0.5*perp[1],
                                 d['label'], fontsize=10, color=d['color'],
                                 ha='center', va='center')

    def plot_simulation_state(self, model, current_state, chosen_edge=None):
        """
        Redraws the model and highlights one edge (chosen_edge) with the same arc offset
        and color that was used in draw_better_edges.
        """
        self.ax.clear()
        self.ax.axis('off')
        G = self.build_graph(model)
        pos = nx.spring_layout(G, seed=0)

        # Mark the current state in red, others in lightgray
        node_colors = ['red' if n == current_state else 'lightgray' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)

        # Draw the normal edges first
        self.draw_better_edges(G, pos)

        # Now highlight the chosen edge with the same curvature offset
        if chosen_edge is not None:
            (t_type, dep, act, dest) = chosen_edge
            if (dep, dest) in G.edges():
                # We replicate the same approach used in draw_better_edges:
                # group edges by (u,v), find the correct one among multiple edges, etc.
                groups = {}
                for u, v, d in G.edges(data=True):
                    groups.setdefault((u, v), []).append(d)

                if (dep, dest) in groups:
                    eds = groups[(dep, dest)]
                    base_offset = 0.33
                    # The offset for each edge among multiple edges from dep->dest
                    offs = [((i - (len(eds) - 1) / 2) * base_offset) for i in range(len(eds))]

                    # If (dep,dest) is also in the reversed set, we do the "adjust" trick
                    opp = (dest, dep) in groups
                    adjust = (opp and dep > dest)

                    # We'll choose the index of the edge that matches t_type in arrow_type
                    # (and optionally the same action, if you want a more precise match).
                    index = 0
                    for i, d in enumerate(eds):
                        # If there's only one, or if we don't care about multiple MDP edges,
                        # we just pick the first that matches arrow_type. 
                        # (If you want to match the action name, you can do so as well.)
                        if d.get('arrow_type') == t_type:
                            index = i
                            break

                    off = offs[index]
                    cur = -off if t_type == "MC" else off
                    if adjust:
                        cur += base_offset if cur >= 0 else -base_offset

                    # Use the same color as the original edge:
                    chosen_color = eds[index].get('color', 'red')  # default to red if missing

                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(dep, dest)],
                        edge_color=chosen_color,
                        width=3,
                        connectionstyle=f'arc3, rad={cur}',
                        ax=self.ax
                    )

        self.draw()

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

        # Control buttons
        self.loadButton = QPushButton("Load Model File")
        self.loadButton.clicked.connect(self.load_file)
        self.exampleButton = QPushButton("Use Example")
        self.exampleButton.clicked.connect(self.load_example)
        self.simulateButton = QPushButton("Launch Simulation!")
        self.simulateButton.setStyleSheet("background-color: lightgreen")
        self.simulateButton.clicked.connect(self.toggle_simulation)
        self.printMatrixButton = QPushButton("Print Matrix")
        self.printMatrixButton.clicked.connect(self.print_matrix)
        self.delayLabel = QLabel("Transition Delay :")
        self.delaySpinBox = QDoubleSpinBox()
        self.delaySpinBox.setRange(0.05, 5.0)
        self.delaySpinBox.setSingleStep(0.05)
        self.delaySpinBox.setValue(0.5)

        # Probability analysis button
        self.probabilityButton = QPushButton("Probability analysis")
        self.probabilityButton.clicked.connect(self.probability_analysis)

        # Layout setup
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.loadButton)
        hlayout.addWidget(self.exampleButton)
        hlayout.addWidget(self.simulateButton)
        hlayout.addWidget(self.printMatrixButton)
        hlayout.addWidget(self.delayLabel)
        hlayout.addWidget(self.delaySpinBox)
        hlayout.addWidget(self.probabilityButton)

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
            self, "Open Model File", "", "MDP Files (*.mdp);;All Files (*)"
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
        text, ok = QInputDialog.getText( self, "Win States", "Enter win state(s), separate using \\',\\' :" )
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
            method_options,
            0,  
            False  # user cannot type a custom item
        )
        if not ok2:
            return 
        if not inc:
            print("\n[PROBA] No incertitude states, no computation needed.")
            return

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
        print(f"[SIM] Starting in state {self.current_state}")
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
        # Gather all possible next states from current
        outgoing = []
        for t in self.model.transitions:
            if t[1] == self.current_state:
                for dest in t[3]:
                    outgoing.append((t[0], t[1], t[2], dest))
        if not outgoing:
            print(f"[SIM] No transitions from {self.current_state}, stopping.")
            self.stop_simulation()
            return
        # if there's exactly one and it's self-loop => infinite loop => stop
        if len(outgoing) == 1 and outgoing[0][3] == self.current_state:
            print(f"[SIM] {self.current_state} is in a loop => stopping.")
            self.stop_simulation()
            return
        # choose randomly among outgoing
        chosen_edge = random.choice(outgoing)
        t_type, dep, act, dest = chosen_edge
        if t_type == "MDP":
            print(f"[MDP] {dep} --{act}--> {dest}")
        else:
            print(f"[MC] {dep} --> {dest}")

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        win.process_file(fname)
    win.show()
    sys.exit(app.exec_())
