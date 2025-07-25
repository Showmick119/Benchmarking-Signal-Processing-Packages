import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Encoder1(nn.Module):
    def __init__(self, in_channels = 1, mid_channels = 32, out_channels = 32, 
                 kernel_size = 5, stride = 1, padding = 2, dilation = 1, 
                 groups = 1, bias = True, pooling = 'max'):
        super(Encoder1, self).__init__()
        self.pooling = pooling.lower()
        # self.encoder = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
        #                         padding, dilation, groups, bias) # bias set to False because batch normalization follows immediately
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            #nn.BatchNorm1d(num_features = mid_channels),
            nn.GELU(),
            nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            #nn.BatchNorm1d(num_features = out_channels),
            nn.GELU()
        )
        
        self.temporal_pool = nn.MaxPool1d(kernel_size = 2, stride = 2)

    def forward(self, x, mask):
        # x: (N, 1, T) , mask: (N, T)
        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)
        B, C, T = x.shape

        x = self.encoder(x)
        mask = F.avg_pool1d(mask, kernel_size = 2, stride = 2)
        x = self.temporal_pool(x)

        if self.pooling == 'max':
            x_masked = x.masked_fill(mask == 0, -torch.inf)
            pooled, _ = x_masked.max(dim=2)
            pooled = torch.nan_to_num(pooled, nan = 0.0, posinf = 0.0, neginf = 0.0)
        else:
            raise ValueError(f"Unknow pooling method: {self.pooling}")
        
        return pooled

        # x = self.conv1d(x)
        # x = self.batch_norm(x)
        # #x = F.relu(x)       # (N, C, T) #TODO: GeLU or LeadyReLU
        # x = F.gelu(x)
        

        # # Masked mean pooling
        # # x_masked = x * mask
        # # summed = x_masked.sum(dim=2)
        # # lengths = mask.sum(dim=2)
        # # mean_pooled = summed / lengths.clamp(min=1e-6)  # (N,c)

        # # Masked max pooling
        # x_masked = x * mask
        # x_masked = x_masked.masked_fill(mask==0, -torch.inf) # Needed for max pooling!
        # #print (x_masked[0][1])
        # max_pooled, _ = x_masked.max(dim=2)  # shape: (batch_size, channels)
        # max_pooled = torch.nan_to_num(max_pooled, nan=0.0, posinf = 0.0, neginf = 0.0)
        
        # return max_pooled
    
class Encoder2(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=32,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 use_input_norm=True,
                 pooling='max',
                 attention_hidden_dim=64):
        super(Encoder2, self).__init__()

        self.use_input_norm = use_input_norm
        self.pooling = pooling.lower()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        if self.pooling == "attention":
            self.attention_net = nn.Sequential(
                nn.Conv1d(out_channels, attention_hidden_dim, kernel_size=1),
                nn.Tanh(),
                nn.Conv1d(attention_hidden_dim, 1, kernel_size=1)
            )

    def forward(self, x, mask):
        # x: (N, 1, T), mask: (N, T)
        x = x.unsqueeze(1)

        if self.use_input_norm:
            # Normalize each segment independently along time
            x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-6)

        x = self.encoder(x)  # (N, C, T)

        # Expand mask to match (N, 1, T) -> broadcast to (N, C, T)
        mask = mask.unsqueeze(1)

        # Apply mask
        x_masked = x * mask

        if self.pooling == "max":
            # Replace 0-masked elements with -inf for max pooling
            x_masked = x_masked.masked_fill(mask == 0, -torch.inf)
            pooled, _ = x_masked.max(dim=2)
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.pooling == "mean":
            summed = x_masked.sum(dim=2)
            lengths = mask.sum(dim=2).clamp(min=1e-6)
            pooled = summed / lengths
        elif self.pooling == "attention":
            attn_scores = self.attention_net(x)
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
            attn_weights = F.softmax(attn_scores, dim=2)
            pooled = torch.sum(x*attn_weights, dim=2)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        print (f'CNN out dim: {pooled.shape}')
        return pooled  # (N, C)


class Encoder3(nn.Module):
    def __init__(self,
                 in_channels=1,
                 mid_channels = 32,
                 out_channels=64,
                 kernel_size=5,
                 stride=1,
                 padding=2,
                 use_input_norm=False,
                 pooling='max',
                 attention_hidden_dim=64,
                 fc_out_dim = 512):
        super(Encoder2, self).__init__()

        self.use_input_norm = use_input_norm
        self.pooling = pooling.lower()
        self.fc_out_dim = fc_out_dim

        self.conv1 = nn.Sequential(
            nn.conv1d(in_channels, mid_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm1d(mid_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.conv1d(mid_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        self.pool = nn.MaxPool1d(kernel_size = 5, stride = 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels, fc_out_dim)
        
    def forward(self, x, mask):
        x = x.unsqueeze(1)
        B, C, T = x.shape

        if self.use_input_norm:
            # Normalize each segment independently along time
            x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-6)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        mask = mask.unsqueeze(1)

        x_masked = x * mask




class BiLSTMRegressor(nn.Module):
    def __init__(self, cnn_out_dim = [32, 64], lstm_hidden_dim = 64, lstm_layers = 1, dropout = 0.3, pooling = 'max', attention_hidden_dim = 64):
        super().__init__()
        self.cnn = Encoder1(mid_channels=cnn_out_dim[0], out_channels=cnn_out_dim[1])
        self.ln_cnn = nn.LayerNorm(normalized_shape=cnn_out_dim[1])

        self.lstm = nn.LSTM(input_size=cnn_out_dim[1], 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_layers, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)

        # FC layers with pre-activation LayerNorm
        self.ln1 = nn.LayerNorm(2*lstm_hidden_dim)
        self.fc1 = nn.Linear(in_features=2*lstm_hidden_dim, out_features=lstm_hidden_dim, bias = False)
        
        self.ln2 = nn.LayerNorm(lstm_hidden_dim)
        self.fc2 = nn.Linear(in_features=lstm_hidden_dim, out_features=lstm_hidden_dim//2, bias = False)

        self.ln3 = nn.LayerNorm(lstm_hidden_dim//2)
        self.fc3 = nn.Linear(in_features=lstm_hidden_dim//2, out_features=lstm_hidden_dim//8)

        self.fc4 = nn.Linear(in_features=lstm_hidden_dim//8, out_features=1)

    def forward(self, batch_ppg, segment_lengths):
        batch_size = len(batch_ppg)
        cnn_features = []
        seq_lengths = []

        for i in range(batch_size):
            segments = batch_ppg[i]     # (num_segs, max_len) i.e. (number of segments, data_points) for one signal with masking 
            seg_lengths = segment_lengths[i]    # e.g. [643, 649, 648, 646, 639, 642, 647, 642, 646, 637, 624, 635] original lengths without masking

            # Create mask for valid time points
            masks = [torch.arange(segments.size(1)).unsqueeze(0) < l for l in seg_lengths]
            masks = torch.cat(masks, dim=0).to(segments.device).float()  # (num_segs, time_len)

            seg_feats = self.cnn(segments, masks)   # (num_segs, cnn_out_dim)
            cnn_features.append(seg_feats)
            seq_lengths.append(len(seg_feats))

        # print (f'CNN Features: {cnn_features}') --> [(num_segs, cnn_out_dim),..., (num_segs, cnn_out_dim)]

        # pad across time dimension (segments)
        padded_seq = pad_sequence(cnn_features, batch_first=True)   # (B, T, D)
        #print(f"[DEBUG] CNN output std: {padded_seq.std().item():.4f}, mean: {padded_seq.mean().item():.4f}")

        #padded_seq = self.ln_cnn(padded_seq)
        lengths = torch.tensor(seq_lengths, dtype=torch.long)

        # Pack and run through BiLSTM
        packed = pack_padded_sequence(padded_seq, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)    # (B, T, 2*H) e.g. (4, 12, 2*64)
        #print(f"[DEBUG] LSTM output std: {lstm_out.std().item():.4f}, mean: {lstm_out.mean().item():.4f}")
        
        # Collect final hidden states based on actual lengths
        final_hidden = []
        for j, l in enumerate(lengths):
            final_hidden.append(lstm_out[j, l-1, :])    # last valid time steps
        final_hidden = torch.stack(final_hidden)    # (B, 2*H)

        # Normalize LSTM output
        #x = self.ln1(final_hidden)
        x = final_hidden

        # FC layers with normalization and activation
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)

        #x = F.relu(self.fc2(self.ln2(x)))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        
        #x = F.relu(self.fc3(self.ln3(x)))s
        x = F.relu(self.fc3(x))
        #print (f"[DEBUG] Before fc4: {x}")
        #print(f"[DEBUG] Final hidden before fc4 std: {x.std().item():.4f}, mean: {x.mean().item():.4f}")
        output = self.fc4(x)    # (B, 1)
        #print (f"[DEBUG] After fc4: {output.squeeze(-1)}")

        return output.squeeze(-1)
    

class FullyConnected(nn.Module):
    def __init__(self, cnn_out_dim = 32, hidden_dims = [32, 16, 1], layers = 3, dropout = 0.3):
        super().__init__()
        assert len(hidden_dims) == layers, "Length of hidden_dims must match number of layers"
        #self.cnn = CNNEncoder(out_channels=cnn_out_dim)
        self.cnn = NewEncoder(out_channels=cnn_out_dim, pooling="attention", attention_hidden_dim=64)
        self.ln_cnn = nn.LayerNorm(normalized_shape=cnn_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fcs = nn.ModuleList()
        input_dim = cnn_out_dim

        for i in range(layers):
            self.fcs.append(nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dims[i], bias = False),
                nn.ReLU() if i < layers -1 else nn.Identity()
            ))
            input_dim = hidden_dims[i]
    
    def forward(self, batch_ppg, segment_lengths):
        batch_size = len(batch_ppg)
        features = []

        for i in range(batch_size):
            segments = batch_ppg[i]
            seg_lengths = segment_lengths[i]

            masks = [torch.arange(segments.size(1)).unsqueeze(0) < l for l in seg_lengths]
            masks = torch.cat(masks, dim=0).to(segments.device).float()
            segments = segments.unsqueeze(1)

            seg_feats = self.cnn(segments, masks)
            #print (torch.var(seg_feats, dim=0))
            mean_feat = seg_feats.max(dim=0).values
            features.append(mean_feat)

        x = torch.stack(features)
        x = self.ln_cnn(x)

        for fc_layer in self.fcs:
            x = fc_layer(x)
            x = self.dropout(x)
        
        return x.squeeze(-1)