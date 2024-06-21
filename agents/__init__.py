# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from bbrl_cl.core import CRLAgents

from .subspace_agents import *


def SubspaceActionAgent(n_initial_anchors, dist_type, refresh_rate, input_dimension,output_dimension, hidden_size, start_steps, resampling_policy, repeat_alpha):
    """ActionAgent that is using "alphas" variable during forward to compute a convex combination of its anchor policies.
    """
    return SubspaceAgents(AlphaAgent(n_initial_anchors, dist_type, refresh_rate, resampling_policy, repeat_alpha),
                          SubspaceAction(n_initial_anchors,input_dimension,output_dimension, hidden_size, start_steps)
                          )

def AlphaTwinCritics(n_anchors, obs_dimension, action_dimension, hidden_size):
    """Twin critics model used for SAC. In addition to the (obs,actions), they also take the convex combination alpha as as input.
    """
    return SubspaceAgents(AlphaCritic(n_anchors, obs_dimension, action_dimension, hidden_size, output_name = "q1"),
                          AlphaCritic(n_anchors, obs_dimension, action_dimension, hidden_size, output_name = "q2")
                          )