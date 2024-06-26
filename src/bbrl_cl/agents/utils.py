# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from bbrl_cl.core import CRLAgent, CRLAgents

class SubspaceAgents(CRLAgents):
    def add_anchor(self, **kwargs):
        for agent in self:
            agent.add_anchor(**kwargs)
    def remove_anchor(self, **kwargs):
        for agent in self:
            agent.remove_anchor(**kwargs)
    def set_best_alpha(self, **kwargs):
        for agent in self:
            agent.set_best_alpha(**kwargs)

class SubspaceAgent(CRLAgent):
    def add_anchor(self, **kwargs):
        pass
    def remove_anchor(self, **kwargs):
        pass
    def set_best_alpha(self, **kwargs):
        pass