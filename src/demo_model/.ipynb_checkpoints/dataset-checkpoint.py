import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from src.base.dataset import CustomDataset


class DemoDataset(CustomDataset):
    def __init__(
        self,
        processor,
        annotation_file,
        image_folder,
        transforms=None,
        ignore_rejected=True,
        ignore_precanned=True,
        test=False,
    ):
        super().__init__(
            annotation_file, image_folder, transforms, ignore_rejected, ignore_precanned
        )
        self.processor = processor
        self.test = test
        if self.test:
            self.image_ids = self.vizwiz.getImgIds()

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(
            self.image_folder, self.vizwiz.loadImgs(img_id)[0]["file_name"]
        )
        img = Image.open(img_path)  # Use PIL to open the image

        # if self.transforms:
        #     img = self.transforms(img)

        if not self.test:
            anns = self.vizwiz.imgToAnns[img_id]
            # caption = anns[0]['caption'] if 'caption' in anns[0] else anns[1]['caption']
            # randomly select a caption
            caption = random.choice(
                [ann["caption"] for ann in anns if "caption" in ann]
            )

            # passing both text and image wont work (Ref: https://github.com/huggingface/transformers/blob/15f8296a9b493eaa0770557fe2e931677fb62e2f/src/transformers/models/git/processing_git.py#L90)
            # get them seperately and combine
            image_encoding = self.processor(
                images=img, padding="max_length", return_tensors="pt"
            )
            text_encoding = self.processor(
                text=caption, padding="max_length", return_tensors="pt"
            )

            encoding = {
                "input_ids": text_encoding["input_ids"].squeeze(),
                "attention_mask": text_encoding["attention_mask"].squeeze(),
                "pixel_values": image_encoding["pixel_values"].squeeze(),
                "image_ids": torch.tensor(img_id),
            }

            return encoding
        else:
            image_encoding = self.processor(
                images=img, padding="max_length", return_tensors="pt"
            )

            encoding = {
                "pixel_values": image_encoding["pixel_values"].squeeze(),
                "image_ids": torch.tensor(img_id),
            }

            return encoding
