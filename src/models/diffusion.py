import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes the time step 't' into a vector so the network knows 
    how much noise is currently present.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LigandDenoisingNetwork(nn.Module):
    """
    The 'Brain' of the diffusion process. 
    Input: Noisy Ligand Coordinates + Time + Protein Context
    Output: Predicted Noise (to be removed)
    """
    def __init__(self, hidden_dim=128, context_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Context adapter (Protein embedding from GNN)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # The Denoising Backbone (ResNet-style MLP for particles)
        self.input_proj = nn.Linear(3, hidden_dim) # 3 for (x,y,z) coords
        
        self.block1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.block2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.block3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        
        self.final_proj = nn.Linear(hidden_dim, 3) # Predict noise in (x,y,z)

    def forward(self, x, t, context):
        """
        x: [batch, num_atoms, 3] (Noisy coordinates)
        t: [batch] (Time step)
        context: [batch, context_dim] (Protein embedding)
        """
        # Embed time and context
        t_emb = self.time_mlp(t)            # [batch, hidden]
        c_emb = self.context_mlp(context)   # [batch, hidden]
        
        # Expand for atoms: [batch, 1, hidden]
        t_emb = t_emb.unsqueeze(1)
        c_emb = c_emb.unsqueeze(1)

        # Process input
        h = self.input_proj(x) # [batch, atoms, hidden]
        
        # Add conditioning (Shift and Scale)
        h = h + t_emb + c_emb 
        
        h = self.block1(h) + h # Residual connection
        h = self.block2(h) + h
        h = self.block3(h) + h
        
        return self.final_proj(h)

class BioDiffusion(nn.Module):
    """
    Main Diffusion Wrapper. Handles the forward noise scheduling 
    and the reverse sampling loop.
    """
    def __init__(self, hidden_dim=128, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # The Network
        self.denoise_net = LigandDenoisingNetwork(hidden_dim=hidden_dim)

        # --- Noise Schedule (Linear Beta Schedule) ---
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-calculate diffusion variables
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        # Register as buffers (part of state_dict but not trainable params)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward Pass: Add noise to data at timestep t
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def p_losses(self, x_start, t, context, noise=None):
        """
        Training Loss Calculation
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 1. Corrupt the data
        x_noisy, noise = self.q_sample(x_start, t, noise)
        
        # 2. Predict the noise using the network
        predicted_noise = self.denoise_net(x_noisy, t, context)
        
        # 3. Compare actual noise vs predicted noise (MSE Loss)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples, num_atoms, context, device):
        """
        Reverse Process: Generate new molecules from pure noise
        """
        # Start with pure random noise
        img = torch.randn((num_samples, num_atoms, 3), device=device)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.denoise_net(img, t, context)
            
            # Remove noise step (Simplified DDPM sampling)
            alpha = 1 - self.betas[i]
            alpha_hat = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (beta / sqrt(1-alpha_hat)) * epsilon) + sigma * z
            img = (1 / torch.sqrt(alpha)) * (img - ((beta / torch.sqrt(1 - alpha_hat)) * pred_noise))
            img = img + torch.sqrt(beta) * noise
            
        return img

if __name__ == "__main__":
    print("🌫️ Initializing Diffusion Model...")
    model = BioDiffusion()
    
    # Mock Data
    batch_size = 2
    atoms = 10
    x_real = torch.randn(batch_size, atoms, 3) # Dummy coordinates
    t = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 128) # Dummy protein embedding
    
    loss = model.p_losses(x_real, t, context)
    print(f"✅ Forward Pass Test - Loss: {loss.item()}")
    
    generated = model.sample(batch_size, atoms, context, "cpu")
    print(f"✅ Sampling Test - Generated Shape: {generated.shape}")