import sys
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import networkx as nx

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ANTLR et imports locaux du parser
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from gramLexer import gramLexer
from gramParser import gramParser
from gramListener import gramListener
from colorama import Fore, Style, init as colorama_init

# Fichier exemple par défaut.
DEFAULT_EXAMPLE_FILE = "ex.mdp"

# --- Le Listener (inchangé) ---
class gramPrintListener(gramListener):
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
        for t in self.transitions:
            if t[2] and t[2] not in self.actions:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Action {Fore.YELLOW}{t[2]}{Style.RESET_ALL} not defined!")
            if t[1] not in self.states:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} State {Fore.YELLOW}{t[1]}{Style.RESET_ALL} not defined!")
            for s in t[3]:
                if s not in self.states:
                    print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} state {Fore.YELLOW}{s}{Style.RESET_ALL} not defined!")
            if any(z < 0 for z in t[4]):
                print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Negative weights detected in transition {Fore.LIGHTRED_EX}{t}{Style.RESET_ALL}")
            if t[0] == "MC":
                for other in self.transitions:
                    if other[0] == "MDP" and t[1] == other[1]:
                        print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Cannot have MDP and MC transitions in the same state {Fore.LIGHTRED_EX}({t[1]}){Style.RESET_ALL}")
                        valid = False
        return valid

def check(model, model_name):
    print(Fore.LIGHTBLUE_EX + f"\nModel: {model_name}" + Style.RESET_ALL)
    model.describe()
    print()
    if not model.validate():
        print(Fore.LIGHTRED_EX + "Model is not valid! Quitting ...\n" + Style.RESET_ALL)
        sys.exit(1)
    print(Fore.LIGHTGREEN_EX + "Model is valid!\n" + Style.RESET_ALL)

# --- Canvas de tracé ---
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.show_welcome()

    def show_welcome(self):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.text(0.5, 0.5, "Bienvenue au simulateur MC/MDP",
                     ha='center', va='center', transform=self.ax.transAxes, fontsize=16)
        self.draw()

    def plot_model(self, model):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Construction d'un MultiDiGraph avec toutes les arêtes.
        G = nx.MultiDiGraph()
        for t in model.transitions:
            trans_type, dep, action, dest_states, weights = t
            total = sum(weights)
            probs = [w / total for w in weights]
            arrow_type = "MDP" if action else "MC"  # Détermine le type d'arc
            for i, dest in enumerate(dest_states):
                label = f"[{action}]\n{probs[i]:.2f}" if action else f"{probs[i]:.2f}"
                col = 'blue' if action else 'red'
                G.add_edge(dep, dest, label=label, color=col, arrow_type=arrow_type)
        
        pos = nx.spring_layout(G, seed=0)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        
        # Regroupement par paire orientée : clé = (u,v)
        groups = {}
        for u, v, data in ((u, v, d) for u, v, d in G.edges(data=True)):
            groups.setdefault((u,v), []).append(data)
        
        base_offset = 1/3
        # Identifier les groupes opposés
        opposite = {}
        for (u,v) in groups:
            if (v,u) in groups:
                opposite[(u,v)] = True
                opposite[(v,u)] = True
        
        for (u,v), data_list in groups.items():
            n = len(data_list)
            if n % 2 == 1:
                offsets = [(i - n//2) * base_offset for i in range(n)]
            else:
                offsets = [(i - n/2 + 0.5) * base_offset for i in range(n)]
            # Conserver l'ordre d'insertion (sans tri)
            # Si le groupe a un opposé et que (u,v) est "secondaire" (u > v), on ajoute un décalage supplémentaire
            adjust = ((u,v) in opposite) and (u > v)
            for offset, data in zip(offsets, data_list):
                # Inverse l'offset pour les arcs MC
                cur_offset = -offset if data.get('arrow_type') == "MC" else offset
                if adjust:
                    cur_offset = cur_offset + (base_offset if cur_offset >= 0 else -base_offset)
                if u == v:
                    loop_rad = 0.2 + abs(cur_offset)
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        edge_color=[data['color']],
                        connectionstyle=f'arc3, rad={loop_rad}',
                        ax=self.ax
                    )
                    x, y = pos[u]
                    self.ax.text(x + cur_offset, y + 0.15, data['label'],
                                 fontsize=10, color=data['color'],
                                 ha='center', va='center',
                                 bbox=dict(facecolor='none', edgecolor='none', pad=0))
                else:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        edge_color=[data['color']],
                        connectionstyle=f'arc3, rad={cur_offset}',
                        ax=self.ax
                    )
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1
                    L = (dx**2 + dy**2)**0.5 if (dx, dy) != (0,0) else 1
                    # Opération miroir : utiliser le vecteur perpendiculaire (dy/L, -dx/L)
                    perp = (dy / L, -dx / L)
                    lx = xm + cur_offset * 0.5 * perp[0]
                    ly = ym + cur_offset * 0.5 * perp[1]
                    self.ax.text(lx, ly, data['label'], fontsize=10, color=data['color'],
                                 ha='center', va='center',
                                 bbox=dict(facecolor='none', edgecolor='none', pad=0))
        
        self.draw()

# --- Fenêtre principale ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MDP/MC Model Visualizer")
        self.resize(800, 600)
        colorama_init()

        self.canvas = PlotCanvas(self, width=10, height=6, dpi=100)
        self.loadButton = QPushButton("Load Model File")
        self.loadButton.clicked.connect(self.load_file)
        self.exampleButton = QPushButton("Use Example")
        self.exampleButton.clicked.connect(self.load_example)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.loadButton)
        buttonLayout.addWidget(self.exampleButton)

        centralWidget = QWidget()
        layout = QVBoxLayout(centralWidget)
        layout.addLayout(buttonLayout)
        layout.addWidget(self.canvas)
        self.setCentralWidget(centralWidget)

    def load_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "",
            "MDP Files (*.mdp);;All Files (*)", options=options
        )
        if fileName:
            self.process_file(fileName)

    def load_example(self):
        self.process_file(DEFAULT_EXAMPLE_FILE)

    def process_file(self, fileName):
        try:
            with open(fileName, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Error opening file {fileName}: {e}")
            return

        input_stream = InputStream(content)
        lexer = gramLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = gramParser(token_stream)
        tree = parser.program()
        printer = gramPrintListener()
        walker = ParseTreeWalker()
        walker.walk(printer, tree)

        check(printer, fileName)
        self.canvas.plot_model(printer)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
