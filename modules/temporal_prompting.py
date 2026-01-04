import torch
import torch.nn as nn

def get_TemporalPrompt(args):
    if args.temporal_prompt in ['group2-2']:
        return TemporalPrompt_3(args=args)
   

class TemporalPrompt_3(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        kernel_spatial = 11
        kernel_temporal = 3
        kernel = (kernel_temporal, kernel_spatial, kernel_spatial)
        padding = (int((kernel_temporal-1)/2), int((kernel_spatial-1)/2), int((kernel_spatial-1)/2))
        hid_dim_1 = 9
        hid_dim_2 = 9
        hid_dim_3 = 16
        hid_dim_l1 = 16
        
        self.Conv = nn.Sequential(
            nn.Conv3d(3, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.Conv3d(hid_dim_1, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_1, hid_dim_2, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_2, hid_dim_3, kernel_size=kernel, stride=1, padding=padding),
        )
        self.MLP = nn.Sequential(
            nn.Linear(hid_dim_3, hid_dim_l1),
            nn.PReLU(),
            nn.Linear(hid_dim_l1, 3),
        )
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        
        padding_tmp = (int((5-1)/2), int((11-1)/2), int((11-1)/2))
        self.Cal_Net = nn.Conv3d(3, 1, kernel_size=(5, 11, 11), stride=1, padding=padding_tmp)
        self.eta = 6
        
        # For attention-based discriminative region identification
        use_attn = getattr(args, 'use_attn_for_discriminative', False) if args else False
        if isinstance(use_attn, int):
            self.use_attn_for_discriminative = bool(use_attn)
        else:
            self.use_attn_for_discriminative = use_attn
        self.attn_weights = None
        self.attn_alpha = getattr(args, 'attn_alpha', 0.5) if args else 0.5  # Weight for combining attention and temporal variation
        
        self.InterFramePrompt = self.init_InterFramePrompt(args)
        
    def forward(self, x):
        B, T, C, W, H = x.shape
        prompt = x.permute(0, 2, 1, 3, 4)
        prompt = self.Conv(prompt)
        prompt = self.dropout1(prompt)
        prompt = prompt.permute(0, 2, 3, 4, 1)
        prompt = self.MLP(prompt)
        prompt = self.dropout2(prompt)
        prompt = prompt.permute(0, 1, 4, 2, 3)
        prompt = self.dropout2(prompt)
        mask = self.get_mask(x)
        return x + prompt*0.05 + prompt*mask*0.05
    
    def update_attention_weights(self, attn_weights_list, B, T):
        """
        Update attention weights for discriminative region identification.
        
        According to the paper: "By combining this information with intra-frame attention weights"
        We need to extract CLS token's attention weights to patch tokens within each frame.
        This represents how CLS token attends to patch tokens within each frame.
        
        Args:
            attn_weights_list: List of attention weights from transformer layers
            B: Batch size
            T: Number of video frames
        """
        if not self.use_attn_for_discriminative or not attn_weights_list:
            return
        
        # Use only the first layer's attention weights (before prompt injection)
        # According to paper: extract CLS token's attention weights from pre-trained model
        # The first layer returns CLS tokens' attention weights
        # Shape: [T, num_tokens_per_frame * T] (after averaging heads) - optimized version
        # or [num_tokens_per_frame * T, num_tokens_per_frame * T] (full self-attention matrix)
        
        first_layer_attn = None
        
        if len(attn_weights_list) > 0:
            layer_attn_weights = attn_weights_list[0]  # First layer only
            
            # Process first layer attention weights
            if isinstance(layer_attn_weights, list):
                # In visual mode, we get multiple attention outputs
                # The first one should be CLS tokens' attention
                for attn_w in layer_attn_weights:
                    if attn_w is not None:
                        # Average over heads if needed
                        if attn_w.dim() == 3:  # [num_heads, query_len, key_len]
                            attn_w = attn_w.mean(dim=0)  # Average over heads -> [query_len, key_len]
                        
                        # Check if it's CLS tokens' attention [T, num_tokens_per_frame * T]
                        # or full self-attention matrix [num_tokens_per_frame * T, num_tokens_per_frame * T]
                        if attn_w.size(0) == T:
                            # Optimized version: CLS tokens' attention [T, num_tokens_per_frame * T]
                            first_layer_attn = attn_w
                            break
                        elif attn_w.size(0) == attn_w.size(1):
                            # Full self-attention matrix, we'll extract CLS tokens' attention later
                            if first_layer_attn is None or attn_w.size(0) > (first_layer_attn.size(0) if first_layer_attn is not None else 0):
                                first_layer_attn = attn_w
            else:
                if layer_attn_weights is not None:
                    if layer_attn_weights.dim() == 3:
                        layer_attn_weights = layer_attn_weights.mean(dim=0)  # [query_len, key_len]
                    # Check if it's CLS tokens' attention or full self-attention matrix
                    if layer_attn_weights.size(0) == T:
                        # Optimized version: CLS tokens' attention
                        first_layer_attn = layer_attn_weights
                    elif layer_attn_weights.size(0) == layer_attn_weights.size(1):
                        # Full self-attention matrix
                        first_layer_attn = layer_attn_weights
        
        if first_layer_attn is not None:
            self.attn_weights = first_layer_attn
        else:
            self.attn_weights = None
    
    def _extract_spatial_attention_map(self, attn_weights, B, T, patch_size=16, img_size=224):
        """
        Extract spatial attention map from CLS token's attention weights.
        
        According to the paper: extract attention from CLS token to patch tokens
        within each frame. This represents the model's attention to different spatial regions.
        
        Args:
            attn_weights: CLS token's attention weights [T, num_tokens_per_frame * T]
                         or [num_tokens_per_frame * T, num_tokens_per_frame * T] (full self-attention matrix)
            B: Batch size
            T: Number of video frames
            patch_size: Patch size (default 16)
            img_size: Image size (default 224)
        
        Returns:
            spatial_attn_maps: [T, img_size, img_size] spatial attention maps for each frame
        """
        if attn_weights is None:
            return None
        
        num_patches_per_frame = (img_size // patch_size) ** 2  # 49 for 224x224 with patch_size=16
        grid_size = img_size // patch_size  # 14 for 224x224 with patch_size=16
        
        # According to CLIP structure: each frame has 1 CLS + 49 patches + 4 other tokens = 54 tokens
        num_tokens_per_frame = 54  # 1 CLS + 49 patches + 4 other tokens
        
        # Check attention weights shape
        # If it's [T, num_tokens_per_frame * T], it's already CLS tokens' attention
        # If it's [num_tokens_per_frame * T, num_tokens_per_frame * T], it's full self-attention matrix
        if attn_weights.dim() == 2:
            if attn_weights.size(0) == T:
                # Shape: [T, num_tokens_per_frame * T] - CLS tokens' attention (optimized version)
                cls_attn_weights = attn_weights
            elif attn_weights.size(0) == attn_weights.size(1):
                # Shape: [num_tokens_per_frame * T, num_tokens_per_frame * T] - full self-attention matrix
                # Extract CLS tokens' attention from the matrix
                cls_indices = [t * num_tokens_per_frame for t in range(T)]
                cls_attn_weights = attn_weights[cls_indices, :]  # [T, num_tokens_per_frame * T]
            else:
                return None
        else:
            return None
        
        # Extract attention from CLS token to patch tokens for each frame
        spatial_attn_maps = []
        
        for t in range(T):
            # CLS token's attention for frame t
            cls_attn_frame = cls_attn_weights[t, :]  # [num_tokens_per_frame * T]
            
            # For frame t, patches are at indices: t * num_tokens_per_frame + 1 to t * num_tokens_per_frame + 1 + 49
            patch_start_idx = t * num_tokens_per_frame + 1  # Skip CLS token
            patch_end_idx = patch_start_idx + num_patches_per_frame
            
            if patch_end_idx <= cls_attn_frame.size(0):
                # Extract attention from CLS to patches (intra-frame attention)
                frame_attn = cls_attn_frame[patch_start_idx:patch_end_idx]  # [num_patches]
                
                # Reshape to spatial map [grid_size, grid_size]
                if frame_attn.numel() == num_patches_per_frame:
                    spatial_map = frame_attn.reshape(grid_size, grid_size)
                    
                    # Upsample to original image size
                    spatial_map = torch.nn.functional.interpolate(
                        spatial_map.unsqueeze(0).unsqueeze(0),
                        size=(img_size, img_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    spatial_attn_maps.append(spatial_map)
        
        if spatial_attn_maps:
            # Stack: [T, H, W]
            return torch.stack(spatial_attn_maps, dim=0)
        return None
    
    def get_mask(self, x):
        B, T, C, W, H = x.shape
        self.B, self.T = B, T
        
        # Calculate temporal variation mask (original method)
        x_temporal = x.permute(0, 2, 1, 3, 4)
        x_temporal = self.Cal_Net(x_temporal)
        x_temporal = x_temporal.squeeze(1)
        x_temporal = x_temporal.reshape(B, T, 32, 7, 32, 7)
        x_temporal = x_temporal.mean(dim=(2, 4))
        bar = x_temporal.reshape(B, T, -1)
        bar = bar.sort(dim=2, descending=True)[0]
        bar = bar[:, :, self.eta]
        temporal_mask = x_temporal > bar.unsqueeze(2).unsqueeze(3)
        
        # Combine with attention-based mask if available
        if self.use_attn_for_discriminative and self.attn_weights is not None:
            # Extract spatial attention map from attention weights
            spatial_attn_map = self._extract_spatial_attention_map(
                self.attn_weights, B, T, patch_size=16, img_size=224
            )
            
            if spatial_attn_map is not None:
                # Reshape spatial_attn_map to match temporal_mask shape
                # spatial_attn_map: [T, 224, 224]
                # Downsample to match temporal_mask: [B, T, 7, 7]
                spatial_attn_map = spatial_attn_map.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, 224, 224]
                spatial_attn_map = torch.nn.functional.avg_pool2d(
                    spatial_attn_map.view(B * T, 1, 224, 224),
                    kernel_size=32, stride=32
                ).view(B, T, 7, 7)
                
                # Normalize attention map
                spatial_attn_map = (spatial_attn_map - spatial_attn_map.min()) / (
                    spatial_attn_map.max() - spatial_attn_map.min() + 1e-8
                )
                
                # Threshold attention map
                attn_threshold = spatial_attn_map.reshape(B, T, -1).sort(dim=2, descending=True)[0][:, :, self.eta]
                attention_mask = spatial_attn_map > attn_threshold.unsqueeze(2).unsqueeze(3)
                
                # Combine temporal variation and attention masks
                # Use weighted combination: alpha * attention + (1-alpha) * temporal
                combined_mask = self.attn_alpha * attention_mask.float() + (1 - self.attn_alpha) * temporal_mask.float()
                combined_mask = combined_mask > 0.5
                self.mask = combined_mask
            else:
                self.mask = temporal_mask
        else:
            self.mask = temporal_mask
        
        # Reshape mask to original image size
        mask_out = self.mask.unsqueeze(2).unsqueeze(4)
        mask_out = mask_out.repeat(1, 1, 32, 1, 32, 1)
        mask_out = mask_out.reshape(B, T, 224, 224)
        mask_out = mask_out.unsqueeze(2)
        mask_out = mask_out.repeat(1, 1, 3, 1, 1)
        return mask_out
    
    def init_InterFramePrompt(self, args):
        self.Attention = nn.MultiheadAttention(embed_dim = 768, num_heads = 12)
        kernel_temporal = 3 
        kernel_token = 15
        kernel_hid = 25
        kernel = (kernel_temporal, kernel_token, kernel_hid)
        padding = (int((kernel_temporal-1)/2), int((kernel_token-1)/2), int((kernel_hid-1)/2))
        hid_dim_1 = 9
        hid_dim_l1 = 16
        
        self.InterConv = nn.Sequential(
            nn.Conv3d(1, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_1, 1, kernel_size=kernel, stride=1, padding=padding),
        )
        self.InterMLP = nn.Sequential(
            nn.Linear(49, 16),
            nn.PReLU(),
            nn.Linear(16, 4),
        )
        padding_tmp = (int((5-1)/2), int((11-1)/2), int((11-1)/2))
        self.Cal_Net_Inter = nn.Conv3d(1, 1, kernel_size=(5, 11, 11), stride=1, padding=padding_tmp)
        
    def get_mask_Inter(self, x):
        x = self.Cal_Net_Inter(x)
        x = x.squeeze(1)
        mask_tmp = self.mask.reshape(self.B, self.T, -1)
        mask_tmp = mask_tmp.unsqueeze(3)
        x = x + x*mask_tmp
        x = x.mean(dim=(2, 3))
        x = (x-x.min())/(x.max()-x.min())
        return x

    
    def get_inter_frame_prompt(self, x):
        BT, L, D = x.shape
        x = x.reshape(self.B, self.T, L, D)
        x = x[:,:,1:,:]
        x = x.reshape(self.B, -1, D)
        x = x.permute(1, 0, 2)
        x = self.Attention(x, x, x)[0]
        x = x.permute(1, 0, 2)
        x = x.reshape(self.B, self.T, -1, D)
        x = x.unsqueeze(1)
        mask = self.get_mask_Inter(x)
        x = self.InterConv(x)
        x = x.squeeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.InterMLP(x)
        x = x.permute(0, 1, 3, 2)
        mask = mask.unsqueeze(2).unsqueeze(3)
        x = x+x*mask
        x = x.reshape(self.B, -1, 768)
        x = x.unsqueeze(0)
        x = x.repeat(12, 1, 1, 1)
        return x*0.05
    
    
    
