from .pipeline_baseline import Pipeline as P0
from .pipeline_phase_a import Pipeline as P1
from .pipeline_phase_b import Pipeline as P2
from .pipeline_plateau_projection import Pipeline as P3
from .pipeline_penalty import Pipeline as P4
from .pipeline_raja import Pipeline as P5

Pipeline_dict = {
    0: P0,  # PipelineBaseline
    1: P1,  # PipelinePhaseA
    2: P2,  # PipelinePhaseB
    3: P3,  # PipelinePlateauProjection (primary_model=1 or 2 via config)
    4: P4,  # PipelinePenalty
    5: P5,  # PipelineRaja
}
