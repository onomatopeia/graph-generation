from args import Args
from create_graphs import create

args = Args()
args.graph_type = 'NFHS'
output = create(args)
print(len(output))
# assert len(output) == 11164

