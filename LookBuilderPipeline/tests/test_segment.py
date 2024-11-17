import pytest
import torch
from diffusers.utils import load_image
from LookBuilderPipeline.segment import segment_image
from PIL import Image
import numpy as np

class TestSegment:
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.size=512

    def test_segment_image_returns_tuple(self):
        result = segment_image(self,"LookBuilderPipeline/img/p09.jpg")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_segment_image_output_size(self):
        img=load_image("LookBuilderPipeline/img/p09.jpg")
        segmented_outfit, mask, mask_array = segment_image(self,"LookBuilderPipeline/img/p09.jpg")
        assert segmented_outfit.size == img.size
        assert mask_array.T.shape == img.size
        assert mask_array.dtype == np.uint8

    ## deprecated
    # def test_segment_image_invalid_path():
    #     with pytest.raises(Exception):
    #         segment_image(self,"non_existent_image.jpg")

    # @pytest.mark.parametrize("additional_option", [None, "shoe", "bag"])
    # def test_segment_image_various_options(additional_option):
    #     segmented_outfit, mask, mask_array = segment_image("LookBuilderPipeline/img/p09.jpg", additional_option=additional_option)
    #     assert isinstance(segmented_outfit, Image.Image)
    #     assert isinstance(mask_array, np.ndarray)

    def test_segment_image_consistency(self):
        result1 = segment_image(self,"LookBuilderPipeline/img/p09.jpg")
        result2 = segment_image(self,"LookBuilderPipeline/img/p09.jpg")
        np.testing.assert_array_equal(result1[1], result2[1])

    # @pytest.mark.parametrize("size", [(512)])
    def test_segment_image_different_sizes(self):
        segmented_outfit, mask, mask_array = segment_image(self,"LookBuilderPipeline/img/p09.jpg", resize=True,size=self.size)#,aspect_ratio=None,square=False)
        assert segmented_outfit.size[0] == self.size
