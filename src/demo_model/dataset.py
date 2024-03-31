# dataset.py (DemoDataset class with caption processing)
from torchvision import transforms
import torch
from PIL import Image
from ..base.dataset import CustomDataset
import os

class DemoDataset(CustomDataset):
    def __init__(self, annotation_file, image_folder, feature_extractor, tokenizer, transforms=None):
        super().__init__(annotation_file, image_folder)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_folder, self.vizwiz.loadImgs(img_id)[0]["file_name"])
        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        # Process the image with the feature_extractor
        pixel_values = self.feature_extractor(images=img, return_tensors="pt").pixel_values.squeeze()

        # Get caption for the image
        caption = self._get_caption_by_image_id(img_id)

        # Tokenize the caption
        caption_encoding = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        return {
            "pixel_values": pixel_values,
            "input_ids": caption_encoding["input_ids"].squeeze(),
            "attention_mask": caption_encoding["attention_mask"].squeeze(),
            "image_ids": torch.tensor([img_id])
        }
