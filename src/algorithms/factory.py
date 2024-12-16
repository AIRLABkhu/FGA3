from algorithms.sac import SAC
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea_c import SVEA_C
from algorithms.pad import PAD
from algorithms.soda import SODA

algorithm = {
	'sac': SAC,
	'soda': SODA,
	'drq': DrQ,
	'svea_c':SVEA_C,
	'pad':PAD,
	'soda':SODA,
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
