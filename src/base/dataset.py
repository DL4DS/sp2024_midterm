import torch
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
from PIL import Image

try:
    from .vizwiz_api import VizWiz
except:
    from vizwiz_api import VizWiz

from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        image_folder,
        transforms=None,
        ignore_rejected=True,
        ignore_precanned=True,
    ):
        """
        Initialize the Dataset with the path to the annotations file and the image folder.

        :param annotation_file: Path to the VizWiz annotations file.
        :param image_folder: Directory containing the images.
        :param ignore_rejected: Whether to ignore rejected annotations.
        :param ignore_precanned: Whether to ignore precanned (not genuine) annotations.
        """
        self.vizwiz = VizWiz(
            annotation_file=annotation_file,
            ignore_rejected=ignore_rejected,
            ignore_precanned=ignore_precanned,
        )
        self.image_folder = image_folder
        self.image_ids = self._get_valid_image_ids()
        self.transforms = transforms

    def _get_caption_by_image_id(self, img_id):
        """
        Return the caption associated with the specified image ID.

        :param img_id: Image ID.
        :return: A string.
        """
        ann_ids = self.vizwiz.getAnnIds(imgIds=img_id)
        anns = self.vizwiz.loadAnns(ann_ids)
        captions = [ann["caption"] for ann in anns if "caption" in ann]
        return captions[0] if captions else None

    def _get_all_image_ids(self):
        """
        Return all image IDs in the dataset.

        :return: A list of image IDs.
        """
        return list(self.vizwiz.imgToAnns.keys())

    def _get_all_captions(self):
        """
        Return all captions in the dataset.

        :return: A list of captions.
        """
        captions = []
        for img_id, anns in self.vizwiz.imgToAnns.items():
            captions.extend([ann["caption"] for ann in anns if "caption" in ann])
        return captions

    def _get_valid_image_ids(self):
        """
        Filter and return image IDs that have associated captions.

        :return: A list of image IDs.
        """
        valid_ids = []
        for img_id, anns in self.vizwiz.imgToAnns.items():
            if any(ann.get("caption") for ann in anns):
                valid_ids.append(img_id)
        return valid_ids

    def __len__(self):
        """
        Return the number of items in the dataset.

        :return: Number of images with captions.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Return the image and its caption at the specified index.

        :param idx: Index of the item.
        :return: A tuple (image, caption).
        """
        img_id = self.image_ids[idx]
        img = plt.imread(
            os.path.join(
                self.image_folder, self.vizwiz.loadImgs(img_id)[0]["file_name"]
            )
        )
        if self.transforms:
            img = self.transforms(img)
        ann_ids = self.vizwiz.getAnnIds(imgIds=img_id)
        anns = self.vizwiz.loadAnns(ann_ids)
        captions = [ann["caption"] for ann in anns if "caption" in ann]
        # TODO Migh have to undo these comments
        #tokenized_captions = [
        #    self.captioning_preprocessor.process(caption) for caption in captions
        #]

        # Assuming each image has at least one caption, return the first one.
        # This can be modified to return all captions or a selected one based on specific requirements.
        return img, captions[0]    #, tokenized_captions[0]

    def collate_fn(self, batch):
        # TODO might have to undo these comments
        #imgs, captions, tokens = zip(*batch)
        imgs, captions = zip(*batch)
        imgs = torch.stack(imgs)
        #tokens = [torch.tensor(token_seq, dtype=torch.long) for token_seq in tokens]
        #padded_tokens = torch.nn.utils.rnn.pad_sequence(
        #    tokens, batch_first=True, padding_value=0
        #)
        return imgs, captions   #, padded_tokens


if __name__ == "__main__":
    print("Running dataset.py test mode.")

    print("Defining transforms...")
    transforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Do NOT change the dataset path below
    print("Initializing datasaet object...")
    dataset = CustomDataset(
        annotation_file="/projectnb/ds598/materials/datasets/vizwiz/captions/annotations/train.json",
        image_folder="/projectnb/ds598/materials/datasets/vizwiz/captions/train",
        transforms=transforms,
    )

    print ("Length of dataset: ", len(dataset))
    print("Getting the first item...")
    # img, caption, tokens = dataset[1]  # Get the first image and its caption
    img, caption = dataset[1]  # Get the first image and its caption

    print(img.shape)  # (H, W, C)
    print(caption)  # A string
    #print(tokens)

    data_loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

    #for imgs, captions, tokens in data_loader:
    for imgs, captions in data_loader:
        print(imgs.shape)
        print(captions)
        #print(tokens)
        break

"""
(1632, 1224, 3)
A person is holding a white Samsung phone on a bed.
"""
