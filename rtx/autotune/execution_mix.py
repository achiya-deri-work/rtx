"""Versioned workload contract for composite-operator timing."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MLPExecutionMix:
    """Calls per measured workload, not weights applied to unrelated timings.

    A training pair includes forward and its full backward (including internal
    recomputation). grad_forwards build then discard a graph; inference_forwards
    run without grad. Extra forwards do not imply another backward. This is an
    isolated same-shape workload, not a simulation of nested checkpoint liveness
    or distinct teacher weights. Benchmark those in the owning application.
    """

    training_pairs: int = 1
    grad_forwards: int = 0
    inference_forwards: int = 0

    def __post_init__(self):
        counts = (self.training_pairs, self.grad_forwards, self.inference_forwards)
        if any(type(v) is not int or v < 0 for v in counts) or not any(counts):
            raise ValueError('execution counts must be nonnegative integers with nonzero total')

    def validate_phase(self, phase):
        if phase == 'train':
            if not self.training_pairs:
                raise ValueError('train phase requires at least one training pair')
        elif phase == 'forward':
            if self.training_pairs or self.grad_forwards:
                raise ValueError('forward phase requires inference-only execution')
        else:
            raise ValueError('phase must be forward or train')

    def contract(self):
        return dict(revision=1, order='inference_grad_pairs', **asdict(self))

    def execute(self, forward, inputs, backward, grad_context):
        """Return the last semantic result; keep discarded graphs out of residency."""
        result = None
        for _ in range(self.inference_forwards):
            result = None
            with grad_context(False):
                result = (forward(*inputs),)
        for _ in range(self.grad_forwards):
            result = None
            with grad_context(True):
                result = (forward(*inputs),)
        for _ in range(self.training_pairs):
            result = None
            with grad_context(True):
                output = forward(*inputs)
                result = (output, *backward(output))
                del output
        return result
