from diffusers.utils import load_image
from LookBuilderPipeline.image_models.base_image_model import BaseImageModel

def setUp(self):
    """Set up a BaseImageModel instance for testing."""
    self.pose = load_image("LookBuilderPipeline/img/pose_p09.jpg")
    self.mask = load_image("LookBuilderPipeline/img/seg_p09.jpg")
    self.prompt = "beautiful female model in paris"
    self.model = BaseImageModel(pose=self.pose, mask=self.mask, prompt=self.prompt)

def test_initialization(self):
    """Test that the model initializes with the correct attributes."""
    self.assertEqual(self.model.pose, self.pose)
    self.assertEqual(self.model.mask, self.mask)
    self.assertEqual(self.model.prompt, self.prompt)

def test_generate_image_not_implemented(self):
    """Test that generate_image raises NotImplementedError."""
    with self.assertRaises(NotImplementedError):
        self.model.generate_image()

