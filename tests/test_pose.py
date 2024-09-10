import pytest
from PIL import Image
from pose.pose import detect_pose


def test_detect_pose_returns_image():
    # Test that the function returns a PIL Image
    result = detect_pose("img/p09.jpg")
    assert isinstance(result, Image.Image)


def test_detect_pose_output_size():
    # Test that the output image has the same size as the input
    input_image = Image.new('RGB', (512, 512))
    input_image.save("test_input.jpg")
    result = detect_pose("img/p09.jpg",resize=True)
    assert result.size == (512, 512)


def test_detect_pose_invalid_path():
    # Test that the function raises an exception for invalid file path
    with pytest.raises(Exception):
        detect_pose("non_existent_image.jpg")


@pytest.mark.parametrize("image_format", ["jpg", "png", "bmp"])
def test_detect_pose_different_formats(image_format):
    # Test that the function works with different image formats
    input_image = Image.new('RGB', (50, 50))
    test_file = f"test_image.{image_format}"
    input_image.save(test_file)
    result = detect_pose(test_file)
    assert isinstance(result, Image.Image)


def test_detect_pose_content():
    # Test that the output image is different from the input (i.e., pose detection occurred)
    input_image = Image.new('RGB', (50, 50), color='red')
    input_image.save("test_input.jpg")
    result = detect_pose("img/p09.jpg")
    assert result.getdata() != input_image.getdata()
