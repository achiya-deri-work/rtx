import pytest

from benchmarks.run_precision200m_long import budget_command


def test_long_budget_preserves_recipe_and_inputs():
    original=['python','-m','train','--batch','72','--sequence','512','--steps','2730',
              '--warmup-steps','113','--output','old','--checkpoint-interval','2730',
              '--seed','941','--learning-rate','0.0003','--eval-every','227']
    command,tokens=budget_command(original,600000000,'new')
    assert command[command.index('--steps')+1]=='16276'
    assert tokens==599998464
    assert command[command.index('--warmup-steps')+1]=='113'
    assert command[command.index('--checkpoint-interval')+1]=='2048'
    assert original[original.index('--output')+1]=='old'
    for key in ('--batch','--sequence','--seed','--learning-rate','--eval-every'):
        assert command[command.index(key)+1]==original[original.index(key)+1]
    with pytest.raises(ValueError):budget_command(original,100,'new')
