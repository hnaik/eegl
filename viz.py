# %%
from pathlib import Path
import networkx as nx

from egr import util


filepath = Path('/Users/hnaik/results/iclr-230903/c60/r2/c60/02/explain/id-0001.json')

G = util.load_graph(filepath)

nx.draw(G)
# %%
