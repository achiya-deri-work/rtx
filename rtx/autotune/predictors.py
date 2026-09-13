"""Portable research-head proposals for the shared UCB orchestrator."""
from .._portable_predictor import PortablePredictor, load_predictor
from .core import Proposal


class PortablePredictorSearch:
    def __init__(self, predictor, pool, head):
        self.predictor, self.pool, self.head = predictor, pool, head
        self.name = 'portable_'+head
        self.order = None

    def propose(self, adapter, history, rng, limit):
        if self.order is None:
            self.order = self.predictor.order(adapter, self.pool, self.head)
        return [Proposal(c, self.name, metadata={'predictor_sha256': self.predictor.digest,
                                                 'predictor_head': self.head})
                for c in self.order if adapter.config_id(c) not in history.seen_ids][:limit]

    def observe(self, observation):
        pass


__all__ = ['PortablePredictor', 'PortablePredictorSearch', 'load_predictor']
