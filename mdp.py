from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
from colorama import Fore, Back, Style, init as colorama_init

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
        print(Fore.LIGHTBLUE_EX + "\n-----------------------------------------------------" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + "States: "     + Style.RESET_ALL, self.states)
        print(Fore.LIGHTBLUE_EX + "Actions: "    + Style.RESET_ALL, self.actions)
        print(Fore.LIGHTBLUE_EX + "Transitions:" + Style.RESET_ALL)
        for x in self.transitions:
            print(" -", x) 
        print(Fore.LIGHTBLUE_EX + "-----------------------------------------------------\n" + Style.RESET_ALL)
            
    def validate(self):
        valid = True
        for x in self.transitions:
            # Check if actions are all declared
            if x[2] and x[2] not in self.actions:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} Action {Fore.YELLOW}{x[2]}{Style.RESET_ALL} not defined!")
            # Check if states are all declared
            if x[1] not in self.states:
                print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} State {Fore.YELLOW}{x[1]}{Style.RESET_ALL} not defined!")
            for y in x[3]:
                if y not in self.states:
                    print(f"{Fore.YELLOW}[Warning]{Style.RESET_ALL} state {Fore.YELLOW}{y}{Style.RESET_ALL} not defined!")
            # Check if weights all sum to 1
            if sum(x[4]) != 10:
                print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Weights do not sum to 1 in transition {Fore.LIGHTRED_EX}{x}{Style.RESET_ALL}")
                valid = False
            # Check if there is no MDP and MC transitions in the same state
            if x[0] == "MC":
                for y in self.transitions:
                    if y[0] == "MDP" and x[1] == y[1]:
                        print(f"{Fore.LIGHTRED_EX}[ERROR]{Style.RESET_ALL} Cannot have MDP and MC transitions in the same state {Fore.LIGHTRED_EX}{x[1]}{Style.RESET_ALL}")
                        valid = False
        return valid

def check(model):
    model.describe()
    valid = model.validate()
    if not valid:
        print(Fore.LIGHTRED_EX + "Model is not valid! Quitting ...\n" + Style.RESET_ALL)
        sys.exit(1)
    print(Fore.LIGHTGREEN_EX + "Model is valid!\n" + Style.RESET_ALL)
    

def main():
    colorama_init()

    lexer   = gramLexer(StdinStream())
    stream  = CommonTokenStream(lexer)
    parser  = gramParser(stream)
    tree    = parser.program()
    printer = gramPrintListener()
    walker  = ParseTreeWalker()
    walker.walk(printer, tree)
    
    check(printer)
    
    print(Fore.LIGHTBLUE_EX + "Generating MDP ..." + Style.RESET_ALL)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            sys.stdin = file
            main()
    else:
        print("Usage: python3 mdp.py <filename>")
        
        
