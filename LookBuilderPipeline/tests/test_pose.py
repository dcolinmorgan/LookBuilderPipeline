import pytest
from PIL import Image
from LookBuilderPipeline.pose.pose import detect_pose

def test_detect_pose_returns_image():
    result = detect_pose("LookBuilderPipeline/img/p09.jpg")
    assert isinstance(result, Image.Image)

def test_detect_pose_output_size():
    result = detect_pose("LookBuilderPipeline/img/p09.jpg",resize=True,size=(512,512))
    assert result.size == (512, 512)


def test_detect_pose_invalid_path():
    with pytest.raises(Exception):
        detect_pose("non_existent_image.jpg")


@pytest.mark.parametrize("image_format", ["jpg", "png", "bmp"])
def test_detect_pose_different_formats(image_format):
    input_image = Image.new('RGB', (50, 50))
    test_file = f"test_image.{image_format}"
    input_image.save(test_file)
    result = detect_pose(test_file)
    assert isinstance(result, Image.Image)


def test_detect_pose_content():
    input_image = Image.new('RGB', (50, 50), color='red')
    input_image.save("test_input.jpg")
    result = detect_pose("LookBuilderPipeline/img/p09.jpg")
    assert result.getdata() != input_image.getdata()
