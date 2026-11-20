"""Small expert weight reductions using the native MXFP8 K32 instruction."""
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm120_utils

from ..configs.moe import short_dw_config,short_dw_rejection
from .mxfp8_moe_cooperative import CooperativeExpertGemm


class ShortReductionExpertDW(CooperativeExpertGemm):
    """One/two K32 issues per copy stage; no physical K128 staging.

    Inputs retain the independently quantized leading-axis format. Only
    their placement in shared memory differs. The K_SW32 operand layout
    uses ordinary ldmatrix; E8M0 scales are broadcast by their own tensor
    strides, not a native TMA 128x4 scale tile.
    """
    def __init__(self,problem,config):
        if reason:=short_dw_rejection(problem,config):raise ValueError(reason)
        super().__init__(problem,short_dw_config(config))

    def _setup_static_layouts(self):
        cfg=self.config
        self.tiled_mma=cute.make_tiled_mma(
            self._make_mma_op(),
            cute.make_layout((cfg.atom_layout_m,cfg.atom_layout_n,1)),
            permutation_mnk=sm120_utils.get_permutation_mnk(
                self.tile_shape_mnk,self.sf_vec_size,self.use_mxf8f6f4))
        atom=cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,self.ab_dtype)
        self.a_layout=cute.tile_to_shape(atom,(cfg.tile_m,cfg.tile_k,1),order=(0,1,2))
        self.b_layout=cute.tile_to_shape(atom,(cfg.tile_n,cfg.tile_k,1),order=(0,1,2))
        # Scale partitioning uses logical (rows,K,stages) coordinates. A
        # zero stride across the 32 within-block K coordinates broadcasts
        # exactly one E8M0 byte per source row and reduction block.
        self.sfa_layout=cute.make_layout((cfg.tile_m,(32,cfg.tile_k//32),1),
                                        stride=(1,(0,cfg.tile_m),0))
        self.sfb_layout=cute.make_layout((cfg.tile_n,(32,cfg.tile_k//32),1),
                                        stride=(1,(0,cfg.tile_n),0))
