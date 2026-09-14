"""Portable research-head proposals for the shared UCB orchestrator."""
from .._portable_predictor import PortablePredictor, load_predictor
from .core import Proposal


class PortablePredictorSearch:
    def __init__(self, predictor, pool, head):
        self.predictor, self.pool, self.head = predictor, pool, head
        self.name = 'portable_'+head
        self.order = None
        self._pending_audit = None

    def propose(self, adapter, history, rng, limit):
        if self.order is None:
            self.order = self.predictor.order(adapter, self.pool, self.head)
            self._pending_audit = {
                'predictor_sha256': self.predictor.digest, 'predictor_head': self.head,
                'ordered_config_ids': [adapter.config_id(c) for c in self.order],
            }
        return [Proposal(c, self.name, metadata={'predictor_sha256': self.predictor.digest,
                                                 'predictor_head': self.head,
                                                 'pool_rank': rank})
                for rank, c in enumerate(self.order) if adapter.config_id(c) not in history.seen_ids][:limit]

    def take_audit(self):
        audit, self._pending_audit = self._pending_audit, None
        return audit

    def observe(self, observation):
        pass


__all__ = ['PortablePredictor', 'PortablePredictorSearch', 'load_predictor']
