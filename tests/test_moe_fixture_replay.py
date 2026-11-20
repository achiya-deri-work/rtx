import torch

from benchmarks.benchmark_moe_varlen import initialize_fixture
from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG


def make_fixture(shared,layout='packed',stable=True):
    case=dict(d_model=32,num_experts=8,top_k=4,expert_hidden_dim=64,shared_hidden_dim=shared)
    def build(c):
        return BalancedMoE(c['d_model'],c['num_experts'],c['expert_hidden_dim'],top_k=c['top_k'],
            shared_hidden_features=c['shared_hidden_dim'],device='cpu',config=DEFAULT_MOE_CONFIG,autotune='off')
    return initialize_fixture(build,case,{**case,'expert_hidden_dim':128},[3,5],layout=layout,seed=17391,
                              stable_inputs=stable,stable_experts=stable,device='cpu')


def test_fixture_replay_is_deterministic_in_both_initialization_modes():
    for stable in (False,True):
        a,x,cu,dy=make_fixture(64,stable=stable)
        b,z,cv,dz=make_fixture(64,stable=stable)
        assert torch.equal(x,z) and torch.equal(cu,cv) and torch.equal(dy,dz)
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
        assert x.dtype==dy.dtype==torch.bfloat16 and x.requires_grad


def test_shared_width_changes_do_not_perturb_expert_or_routing_operands():
    a,x,cu,dy=make_fixture(64)
    b,z,cv,dz=make_fixture(128)
    assert torch.equal(x,z) and torch.equal(cu,cv) and torch.equal(dy,dz)
    for left,right in ((a.gate_up,b.gate_up),(a.down,b.down),(a.router,b.router)):
        assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())


def test_flat_replay_is_a_true_global_pool_with_identical_inputs():
    packed,x,cu,dy=make_fixture(64)
    flat,z,none,dz=make_fixture(64,layout='flat')
    assert none is None and cu.tolist()==[0,3,8]
    assert torch.equal(x,z) and torch.equal(dy,dz)
    assert all(torch.equal(v,flat.state_dict()[k]) for k,v in packed.state_dict().items())
