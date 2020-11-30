import unitest
import pytest
from dldp.utils import mask_generation_asap as masap
slide_path = '/home/wli/Downloads/CAMELYON16/training/tumor'
anno_path = '/home/wli/Downloads/CAMELYON16/training/Lesion_annotations'
mask_path = '/home/wli/Downloads/CAMELYON16/masking2'


def test_mask_asap_read_file():
    make_mask = masap.mask_gen_asap(slide_path, anno_path, mask_path)
    assert len(make_mask.anno_paths) == 111
