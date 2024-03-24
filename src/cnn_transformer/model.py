import torch
import torch.nn as nn
from torch import Tensor
import math
from torch.nn import MultiheadAttention, Softmax, Dropout
from torchvision import models
from src.cnn_lstm.model import EncoderCNN, DecoderRNN

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=2, drop_prob=0.3):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout=drop_prob)

        self.mha1 = MultiheadAttention(embed_size, num_heads=num_heads)
        self.mha2 = MultiheadAttention(embed_size, num_heads=num_heads)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

        self.linear = nn.Linear(embed_size, vocab_size)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.dropout = Dropout(drop_prob)

    def forward(self, features, captions):
        # vectorize the captions
        embeds = self.embedding(captions[:, :-1])
        #print("embed size before processing:", embeds.size())
        embeds = self.positional_encoding(embeds)
        embeds = self.dropout(embeds)
        #print("Embeds size after processing:", embeds.size())

        features = torch.cat((features.unsqueeze(1),embeds),dim=1)
        #print("Features size 2:", features.size())
        
        mh1_out, _ = self.mha1(embeds, embeds, embeds)
        #print("output of first mha:", mh1_out.size())

        padding = torch.zeros(mh1_out.shape[0], 1, mh1_out.shape[2], device=mh1_out.device)
        padded_mha1_output = torch.cat([mh1_out, padding], dim=1) # pad the output of mha1 by 1 in the middle dimension
        #print("output of first mha after norm and padding:", padded_mha1_output.size())

        padded_embeds = torch.cat([embeds, padding], dim=1)

        x = self.norm1(padded_embeds + padded_mha1_output)

        mh2_out, _ = self.mha2(padded_mha1_output, features, features)
        x = self.norm2(x + mh2_out)

        x_ = self.ff(x)
        x = self.norm3(x_ + x)
        x = self.linear(x)

        return x

    ### FIX NEEDED - TO WORK WITH BATCHED INPUTS ###
    def generate_caption(self, features, max_len=20, vocab=None):
        # Inference part - given image features, generate caption
        batch_size = features.size(0)
        print("features size", features.size())

        caption_ids = [vocab.stoi["<SOS>"]]
        
        for i in range(max_len):
            caption_tensor = torch.tensor([caption_ids], dtype=torch.long, device="cuda")
            output = self.forward(features, caption_tensor)
            
            predicted_id = output[:, -1, :].argmax(dim=-1).item() # get the token with the highest probability
            caption_ids.append(predicted_id)
            
            if predicted_id == vocab.stoi['<EOS>']:
                break

        # Convert token IDs back to words, excluding the <SOS> and <EOS> tokens for the final caption.
        generated_words = [vocab.itos[idx] for idx in caption_ids if idx not in (vocab.stoi['<SOS>'], vocab.stoi['<EOS>'])]
        caption = ' '.join(generated_words)

        return caption


class CNN_Transformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=1, drop_prob=0.3):
        super(CNN_Transformer, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = Decoder(embed_size, vocab_size, num_heads, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        #print("Feature size from CNN:", features.size())
        outputs = self.decoder(features, captions)
        return outputs
