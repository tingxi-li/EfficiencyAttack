from pipeline_zoo.dual_object_detection_clean import Pipeline as P0
from dual_object_detection_clean_rand import Pipeline as P1
from dual_object_detection_clean_randn import Pipeline as P2
from dual_object_detection import Pipeline as P3
from dual_object_detection_phase_a import Pipeline as P4
from dual_object_detection_phase_b import Pipeline as P5
from dual_object_detection_gradient_projection import Pipeline as P6
from dual_object_detection_penalty import Pipeline as P7

Pipeline_dict = {
    0: P0,
    1: P1,
    2: P2,
    3: P3,
    4: P4,
    5: P5,
    6: P6,
    7: P7
}