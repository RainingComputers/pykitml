from ._minimize_model import MinimizeModel
from ._classifier import Classifier
from .decision_tree import DecisionTree

def _disable_ploting():
    MinimizeModel._plot_graphs=False
    Classifier._plot_graphs=False
    DecisionTree._plot_graphs=False

def _enable_ploting():
    MinimizeModel._plot_graphs=True
    Classifier._plot_graphs=True
    DecisionTree._plot_graphs=True