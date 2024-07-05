# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .agent import SubspaceAgents, AlphaAgent, SubspaceAction, AlphaCritic


def SubspaceActionAgent(n_initial_anchors, dist_type, refresh_rate, input_dimension, output_dimension, hidden_size, start_steps, resampling_policy, repeat_alpha, **kwargs):
    """ActionAgent that is using "alphas" variable during forward to compute a convex combination of its anchor policies.
    """
    return SubspaceAgents(AlphaAgent(n_initial_anchors, dist_type, refresh_rate, resampling_policy, repeat_alpha),
                          SubspaceAction(n_initial_anchors, input_dimension, output_dimension, hidden_size, start_steps)
                          )