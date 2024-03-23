import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Softmax, Dropout
from torchvision import models
from src.cnn_lstm.model import EncoderCNN, DecoderRNN


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads=2, drop_prob=0.3):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.mha1 = MultiheadAttention(embed_size, num_heads=num_heads)
        self.mha2 = MultiheadAttention(embed_size, num_heads=num_heads)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

        self.linear = nn.Linear(embed_size, vocab_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.dropout = Dropout(drop_prob)

    def forward(self, features, captions):
        # vectorize the caption
        embeds = self.embedding(captions[:, :-1])
        mh1_out, _ = self.mha1(embeds, embeds, embeds)
        x = self.norm1(embeds + mh1_out)
        mh2_out, _ = self.mha2(x, features, features)
        x = self.norm2(x + mh2_out)
        x = self.ff(x)
        x = self.norm3(x + x)
        return x

    ### FIX NEEDED - TO WORK WITH BATCHED INPUTS ###
    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = inputs.size(0)

        captions = []

        for i in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())

            # end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            # send generated word as the next caption
            inputs = self.embedding(predicted_word_idx.unsqueeze(0))

        cap = [vocab.itos[idx] for idx in captions]
        cap = ' '.join(cap)
        cap = cap.replace("<EOS>", "")
        cap = cap.replace("<UNK>", "")
        cap = cap.replace("<SOS>", "")
        cap = cap.replace("<PAD>", "")
        cap = cap.strip()
        return cap


class CNN_Transformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(CNN_Transformer, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
