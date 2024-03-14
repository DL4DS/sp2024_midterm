import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from src.base.constants import *
from src.base.helpers import *
from src.base.vizwiz_eval_cap.eval import VizWizEvalCap
from dataset import CNNLSTMDataset  # import from local file dataset.py
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from model import EncoderDecoder

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

### TO TRY - USING GLOVE EMBEDDINGS ###
USE_GLOVE = True
if USE_GLOVE:
    # The directory listed below has the glove embeddings. 
    # These were originally retrieved with
    # `wget http://nlp.stanford.edu/data/glove.6B.zip` and `unzip glove.6B.zip`
    glove_dir = '/projectnb/ds598/materials/misc'
    embeddings_index = {} 
    f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))


if USE_GLOVE:
    CNNLSTM_SAVE_PATH += "_glove"
create_directory(CNNLSTM_SAVE_PATH)  # src/base/helpers.py
create_directory(CNNLSTM_SAVE_PATH + "/examples")



image_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_dataset = CNNLSTMDataset(
    processor=None,
    annotation_file=TRAIN_ANNOTATION_FILE,
    image_folder=TRAIN_IMAGE_FOLDER,
    transforms=image_transforms,
)
val_dataset = CNNLSTMDataset(
    processor=None,
    annotation_file=VAL_ANNOTATION_FILE,
    image_folder=VAL_IMAGE_FOLDER,
    transforms=image_transforms,
)

## Use the Subset while debugging ###
# train_dataset = Subset(train_dataset, range(3000))
# val_dataset = Subset(val_dataset, range(100))

# # ### Since, subset is used above, the dataset object needs to be called with a .dataset, to access the original dataset. So while using the full dataset, the below is done. ###
train_dataset = Subset(train_dataset, range(len(train_dataset)))
val_dataset = Subset(val_dataset, range(len(val_dataset)))


print("SANITY CHECK!!")
print(f"LEN TRAIN IMAGE IDS: {len(train_dataset.dataset.image_ids)}")
print(f"LEN VAL IMAGE IDS: {len(val_dataset.dataset.image_ids)}")
print("SANITY CHECK DONE!!")


def collate_fn(batch):
    images, captions, img_ids = zip(*batch)
    
    # Stack images into a single tensor
    imgs = torch.cat([img.unsqueeze(0) for img in images], dim=0)

    targets = pad_sequence(captions, batch_first=True, padding_value=0)
    img_ids_tensor = torch.tensor(img_ids, dtype=torch.int64) if not isinstance(img_ids[0], torch.Tensor) else torch.stack(img_ids, 0)

    return imgs, targets, img_ids_tensor


train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
) # FIX - make the code work for a larger batch size while testing


device = "cuda" if torch.cuda.is_available() else "cpu"

# get the losses for vizualization
losses = list()
val_losses = list()
total_step = len(train_dataloader)
criterion = nn.CrossEntropyLoss()

vocab_size = len(train_dataset.dataset.vocab)
# move the models to the GPU
model = EncoderDecoder(embed_size=200,hidden_size=512,vocab_size=vocab_size,num_layers=2,drop_prob=0.3).to(device)

if USE_GLOVE:
    ### TO TRY - USING GLOVE EMBEDDINGS ###
    embedding_matrix = np.zeros((vocab_size, 200))
    for i, word in enumerate(train_dataset.dataset.vocab.itos.keys()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(200,))

    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float).to(device)
    model.decoder.embedding.weight = nn.Parameter(embedding_tensor, requires_grad=False)

# Load models
# model.encoder.load_state_dict(torch.load(f"{CNNLSTM_SAVE_PATH}/best_encoder.pth"))
# model.decoder.load_state_dict(torch.load(f"{CNNLSTM_SAVE_PATH}/best_decoder.pth"))

vocab_size = len(train_dataset.dataset.vocab)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



method = "CIDEr"  # method used for comparsions

logger = Logger(f"{CNNLSTM_SAVE_PATH}/logs.log")


def train(logger, train_dataloader, model, optimizer, device):
    model.train()

    for idx, batch in progress_bar:

        images, captions, img_ids = batch
        images,captions = images.to(device),captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs = model(images, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        
        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Update progress bar with loss info
        progress_bar.set_postfix({"loss": loss.item()})

    return loss.item()


def evaluate(
    logger, epoch, save_path, best_score, train_dataset, val_dataloader, model, device
):
    model.eval()
    caption_val = []
    image_ids = []
    plot_captions_dict = {}
    for idx, batch in enumerate(val_dataloader):

        with torch.no_grad():
            image, caption, img_id = batch
            image, caption = image.to(device), caption.to(device)
            features = model.encoder(image.to(device))
            generated_caption = model.decoder.generate_caption(features.unsqueeze(0),vocab=train_dataset.dataset.vocab)

            caption_val.append(
                {"image_id": img_id.item(), "caption": generated_caption}
            )  # Used for VizWizEvalCap
            plot_captions_dict[img_id.item()] = generated_caption  # Used for plotting

    # Save the generated captions to a json file
    with open(f"{save_path}/generated_captions.json", "w") as f:
        json.dump(caption_val, f, indent=4)

    vizwizRes = val_dataset.dataset.vizwiz.loadRes(
        f"{save_path}/generated_captions.json"
    )

    vizwizEval = VizWizEvalCap(val_dataset.dataset.vizwiz, vizwizRes)
    vizwizEval.evaluate()

    logger.info(f"Validation scores at epoch: {epoch}")
    for method in vizwizEval.eval:
        logger.info(f"  Method: {method}, Score: {vizwizEval.eval[method]:.4f}")

    return vizwizEval, vizwizRes, plot_captions_dict


def get_val_examples(vizwizEval, vizwizRes, plot_captions_dict, epoch, method="CIDEr"):
    # Get 5 best and 5 worst captions every epoch
    # Use first 3 idxs to plot throughout the training

    img_id_scores = vizwizEval.imgToEval

    best_img_ids = sorted(
        img_id_scores, key=lambda x: img_id_scores[x][method], reverse=True
    )[:5]
    worst_img_ids = sorted(img_id_scores, key=lambda x: img_id_scores[x][method])[:5]
    first_3_img_ids = list(img_id_scores.keys())[:3]

    best_img_paths = [
        os.path.join(
            val_dataset.dataset.image_folder,
            val_dataset.dataset.vizwiz.loadImgs(img_id)[0]["file_name"],
        )
        for img_id in best_img_ids
    ]
    worst_img_paths = [
        os.path.join(
            val_dataset.dataset.image_folder,
            val_dataset.dataset.vizwiz.loadImgs(img_id)[0]["file_name"],
        )
        for img_id in worst_img_ids
    ]
    first_3_img_paths = [
        os.path.join(
            val_dataset.dataset.image_folder,
            val_dataset.dataset.vizwiz.loadImgs(img_id)[0]["file_name"],
        )
        for img_id in first_3_img_ids
    ]

    best_img_and_captions = [
        (img_path, plot_captions_dict[img_id], vizwizEval.vizwiz.imgToAnns[img_id])
        for img_path, img_id in zip(best_img_paths, best_img_ids)
    ]  # get img path, generated caption and ground truth caption
    worst_img_and_captions = [
        (img_path, plot_captions_dict[img_id], vizwizEval.vizwiz.imgToAnns[img_id])
        for img_path, img_id in zip(worst_img_paths, worst_img_ids)
    ]
    first_3_img_and_captions = [
        (img_path, plot_captions_dict[img_id], vizwizEval.vizwiz.imgToAnns[img_id])
        for img_path, img_id in zip(first_3_img_paths, first_3_img_ids)
    ]

    # Save the images and captions
    save_image_captions(
        best_img_and_captions, f"{CNNLSTM_SAVE_PATH}/examples/epoch_{epoch}/best/"
    )
    save_image_captions(
        worst_img_and_captions, f"{CNNLSTM_SAVE_PATH}/examples/epoch_{epoch}/worst/"
    )
    save_image_captions(
        first_3_img_and_captions, f"{CNNLSTM_SAVE_PATH}/examples/epoch_{epoch}/first_3/"
    )


best_score = 0
for epoch in range(3):
    print(f"Epoch: {epoch+1}")
    # Wrap the dataloader with tqdm for a progress bar
    progress_bar = tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), desc="Training"
    )

    # # Train the model
    loss = train(logger, train_dataloader, model, optimizer, device)
    logger.info(f"Loss at epoch {epoch}: {loss}")

    # Evaluate the model every 3 epochs
    if epoch % 3 == 0:
        vizwizEval, vizwizRes, plot_captions_dict = evaluate(
            logger,
            epoch,
            CNNLSTM_SAVE_PATH,
            best_score,
            train_dataset, 
            val_dataloader,
            model,
            device,
        )
        score = vizwizEval.eval[method]
        logger.info(f"Score at epoch {epoch}: {score}")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"{CNNLSTM_SAVE_PATH}/best_model.pth")

        get_val_examples(vizwizEval, vizwizRes, plot_captions_dict, epoch, method)
