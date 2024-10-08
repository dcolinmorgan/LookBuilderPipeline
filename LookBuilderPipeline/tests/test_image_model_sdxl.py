import pytest
from unittest.mock import patch, MagicMock
# from LookBuilderPipeline.image_models.image_model_sdxl import ImageModelSDXL  # tricky to test on github without GPU runner

@pytest.fixture
def mock_pipe():
    return MagicMock()

@pytest.fixture
def mock_controlnet_model():
    return MagicMock()

# @patch('LookBuilderPipeline.image_model_sdxl.ControlNetModel_Union')
# @patch('LookBuilderPipeline.image_model_sdxl.StableDiffusionXLControlNetUnionInpaintPipeline')
# @patch('LookBuilderPipeline.image_model_sdxl.snapshot_download')
# def test_image_model_sdxl_init(mock_snapshot_download, mock_pipeline, mock_controlnet, mock_pipe, mock_controlnet_model):
#     mock_controlnet.from_pretrained.return_value = mock_controlnet_model
#     mock_pipeline.from_pretrained.return_value = mock_pipe

#     model = ImageModelSDXL("pose.jpg", "mask.jpg", prompt="test prompt")

#     assert isinstance(model, ImageModelSDXL)
#     mock_snapshot_download.assert_called_once()
#     mock_controlnet.from_pretrained.assert_called_once()
#     mock_pipeline.from_pretrained.assert_called_once()
#     # Replace torch.float16 with a string representation
#     mock_pipe.text_encoder.to.assert_called_once_with("float16")
#     mock_pipe.controlnet.to.assert_called_once_with("float16")
#     mock_pipe.enable_model_cpu_offload.assert_called_once()

# @patch('LookBuilderPipeline.image_model_sdxl.detect_pose')
# @patch('LookBuilderPipeline.image_model_sdxl.segment_image')
# @patch('LookBuilderPipeline.image_model_sdxl.load_image')
# def test_generate_image(mock_load_image, mock_segment_image, mock_detect_pose, mock_pipe):
#     model = ImageModelSDXL("pose.jpg", "mask.jpg", prompt="test prompt")
#     model.pipe = mock_pipe

#     mock_detect_pose.return_value = (MagicMock(), MagicMock())
#     mock_segment_image.side_effect = [(None, MagicMock(), None), (None, MagicMock(), None)]
#     mock_load_image.return_value = MagicMock()
#     mock_pipe.return_value.images = [MagicMock()]

#     result = model.generate_image()

#     assert result is not None
#     mock_detect_pose.assert_called_once()
#     assert mock_segment_image.call_count == 2
#     mock_load_image.assert_called_once()
#     mock_pipe.assert_called_once()

if __name__ == "__main__":
    pytest.main()
