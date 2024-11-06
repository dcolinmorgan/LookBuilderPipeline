import pytest
import torch
from PIL import Image
from LookBuilderPipeline.pose import detect_pose


class TestPose:
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_detect_pose_returns_image(self):
        result = detect_pose(
            self, 
            image_path="LookBuilderPipeline/img/p09.jpg"
        )
        assert isinstance(result, Image.Image)

    def test_detect_pose_output_size(self):
        result = detect_pose(
            self,
            image_path="LookBuilderPipeline/img/p09.jpg",
            resize=True,
            size=(512)
        )
        assert result.size[0] == 512

    def test_detect_pose_invalid_path(self):
        with pytest.raises(Exception):
            detect_pose(self, image_path="non_existent_image.jpg")

    @pytest.mark.parametrize("image_format", ["jpg", "png", "bmp"])
    def test_detect_pose_different_formats(self, image_format):
        input_image = Image.new('RGB', (560, 512))
        test_file = f"test_image.{image_format}"
        input_image.save(test_file)
        result = detect_pose(self, image_path=test_file)
        assert isinstance(result, Image.Image)

    def test_detect_pose_content(self):
        input_image = Image.new('RGB', (560, 560), color='red')
        input_image.save("test_input.jpg")
        result = detect_pose(
            self, 
            image_path="LookBuilderPipeline/img/p09.jpg"
        )
        assert result.getdata() != input_image.getdata()
