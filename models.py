import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

class CNNLSTMRaw(nn.Module):
    def __init__(self, input_channels=12, hidden_size=128, num_classes=2, lstm_layers=2, dropout=0.5):
        super(CNNLSTMRaw, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.attn = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        attn_out = self.attn(out)
        logits = self.fc(self.dropout(attn_out))
        return logits


class CNNLSTMSpec(nn.Module):
    def __init__(self, input_channels=12, hidden_size=128, num_classes=2, lstm_layers=2, dropout=0.5):
        super(CNNLSTMSpec, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)

        self.attn = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (batch, 512, freq', time')
        x = x.squeeze(3)  # Remove the last dimension (freq' = 1), now (batch, 512, time')
        x = x.permute(0, 2, 1)  # (batch, time', 512)
        out, _ = self.lstm(x)
        attn_out = self.attn(out)
        logits = self.fc(self.dropout(attn_out))
        return logits


class ViT(nn.Module):
    def __init__(self, num_classes=2, input_channels=12):
        super().__init__()

        # Load base config and modify input channels
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.num_channels = input_channels

        # Load model with custom config
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", config=config, ignore_mismatched_sizes=True)

        old_proj = self.vit.embeddings.patch_embeddings.projection
        self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding
        )

        # Re-init the new layer
        nn.init.kaiming_normal_(self.vit.embeddings.patch_embeddings.projection.weight, mode='fan_out', nonlinearity='relu')
        if self.vit.embeddings.patch_embeddings.projection.bias is not None:
            nn.init.zeros_(self.vit.embeddings.patch_embeddings.projection.bias)

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits


class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, dropout, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Heart_GPT_Model(nn.Module):
    def __init__(self, vocab_size=101, block_size=500, n_embd=64, n_head=8, n_layer=8, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=self.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



class HeartGPT(nn.Module):
    def __init__(self, num_classes=2, vocab_size=101, block_size=500, n_embd=64, n_head=8, n_layer=8, dropout=0.2):
        super().__init__()
        self.backbone = Heart_GPT_Model(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout)
        self.backbone.load_state_dict(torch.load('models/HeartGPT/pretrained/ECGPT_560k_iters.pth', map_location='cpu'))
        self.backbone.eval()

        self.classifier = nn.Linear(vocab_size, num_classes)

    def forward(self, x):
        with torch.no_grad():
            logits, _ = self.backbone(x)  # shape (B, T, vocab_size)

        pooled = torch.mean(logits, dim=1)
        return self.classifier(pooled)

