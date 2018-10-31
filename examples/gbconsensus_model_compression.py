from framed.io.sbml import *
from framed.cobra import variability,simulation
from framed.convex.subset_reduction import *
from metaconvexpy.utilities.file_utils import read_pickle, pickle_object

CONSENSUS_MODEL_PATH = '/home/skapur/MEOCloud/Projectos/PhDThesis/Material/Models/Consensus/ConsensusModel.xml'

model = load_cbmodel(CONSENSUS_MODEL_PATH)

biomass_rx = "R_biomass_reaction"
drains = [r for r in model.reactions if 'R_EX_' in r] + [biomass_rx]
media = read_pickle('/home/skapur/Workspaces/PyCharm/metaconvexpy/examples/GBconsensus_resources/GBconsensus_media.pkl')

fva_blocked = variability.blocked_reactions(model, constraints=media)


cmodel = generate_reduced_model(model, to_exclude=fva_blocked, to_keep_single=drains)


wt_orig = simulation.FBA(model, objective={biomass_rx:1}, constraints=media)
wt_comp = simulation.FBA(cmodel, objective={biomass_rx:1}, constraints=media)
