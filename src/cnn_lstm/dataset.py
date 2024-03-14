import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
from src.base.dataset import CustomDataset
from src.base.helpers import Vocabulary
import transformers


class CNNLSTMDataset(CustomDataset):
    def __init__(
        self,
        processor,
        annotation_file,
        image_folder,
        transforms=None,
        ignore_rejected=True,
        ignore_precanned=True,
        training=True
    ):
        super().__init__(
            annotation_file, image_folder, transforms, ignore_rejected, ignore_precanned
        )
        self.transforms = transforms
        self.tokenizer = processor
        # self.vocab = self.vizwiz.build_vocab(
        #     tokenizer_eng, freq_threshold=5, stoi=self.stoi, itos=self.itos
        # )
        # self.vocab = self.tokenizer.get_vocab()
        self.training = training
        if self.training:
            self.captions = self._get_all_captions()
            self.vocab = Vocabulary(freq_threshold=5)
            self.vocab.build_vocab(self.captions)
        else:
            self.image_ids = self.vizwiz.getImgIds()

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(
            self.image_folder, self.vizwiz.loadImgs(img_id)[0]["file_name"]
        )
        img = Image.open(img_path)  # Use PIL to open the image

        if self.transforms:
            img = self.transforms(img)

        if not self.training:
            img_id = torch.tensor(img_id)
            return img, img_id

        anns = self.vizwiz.imgToAnns[img_id]
        # caption = anns[0]['caption'] if 'caption' in anns[0] else anns[1]['caption']
        # randomly select a caption
        caption = random.choice([ann["caption"] for ann in anns if "caption" in ann])

        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        # # Convert to tensor
        caption = torch.tensor(caption_vec)
        img_id = torch.tensor(img_id)

        return img, caption, img_id
