from torchvision import transforms
import torch
from dataset import BioSRDDataset
import os 
from torch.utils.data import DataLoader
import deeplay as dl
from Scheduler import LinearScheduler
import time
from datetime import timedelta

## forward Process 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: ",device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    ])

path = "/kaggle/input/datasets/dhiaelhakezzeddini/biosr-res/biosr_dataset/BioSR/Microtubules"

train_dataset = BioSRDDataset(lr_paths=os.path.join(path,"training_wf"),hr_paths=os.path.join(path,"training_gt"),transform=transform)
test_dataset = BioSRDDataset(lr_paths=os.path.join(path,"test_wf","level_09"),hr_paths=os.path.join(path,"test_gt"),transform=transform)


def positional_encoding(t, enc_dim):
    """Encode position information with a sinusoid."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, enc_dim, 2).float() 
    / enc_dim)).to(t.device)
    pos_enc_a = torch.sin(t.repeat(1, enc_dim // 2) * inv_freq) 
    pos_enc_b = torch.cos(t.repeat(1, enc_dim // 2) * inv_freq) 
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1) 
    return pos_enc

pos_emb_dim = 256
attention_unet = dl.AttentionUNet(
    in_channels=2, ## two 1-channel images : the noisy hr and the lr as a condtion 
    channels = [32,64,128],
    base_channels=[256,256],
    channel_attention=[False,False,False],
    out_channels=1,
    position_embedding_dim=pos_emb_dim
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(attention_unet.parameters(), lr=1e-4)

noise_steps = 2000
diffusion = LinearScheduler(
    n_ts=noise_steps, 
    beta_s=1e-6 ,
    beta_e=0.01,
    img_size=128,
    device=device
)
def prepare_data(input_image, target_image, device=device):
    """Prepare data."""
    batch_size = input_image.shape[0]
    input_image = input_image.to(device)
    target_image = target_image.to(device)
    noise = torch.randn_like(target_image).to(device)
    t = torch.randint(low=0, high=noise_steps, size=(batch_size,)).to(device) ## time step
    x_t, noise = diffusion.add_noise(target_image, noise, t,device) 
    x_t = torch.cat((input_image, x_t), dim=1) ## conditioned input to the Attention Uent model
    t = positional_encoding(t.unsqueeze(1), pos_emb_dim) 
    return x_t.to(device), t.to(device), noise.to(device) 

## training loop : 

epochs = 100
train_loss = []
for epoch in range(epochs): 
    start_time = time.time()   
    num_batches = len(train_loader)
    print("\n" + f"Epoch {epoch + 1}/{epochs}" + "\n" + "_" * 10)
    running_loss = 0.0 
    attention_unet.train() 
    for batch_idx, (input_images, target_images) in enumerate(train_loader, start=0):
        x_t, t, noise = prepare_data(input_images, target_images) 
        outputs = attention_unet(x=x_t, t=t) 
        optimizer.zero_grad() 
        loss = criterion(outputs, noise) 
         
        if batch_idx % 200 == 0: 
            print(f"Batch {batch_idx + 1}/{num_batches}: " 
            + f"Train loss: {loss.item():.4f}")
        running_loss += loss.item() 
        loss.backward()
        optimizer.step()
    train_loss.append(running_loss / num_batches) 
    end_time = time.time() 
    print("-" * 10 + "\n" + f"Epoch {epoch + 1}/{epochs} : " + f"Train loss: {train_loss[-1]:.4f}, " + f"Time taken: {timedelta(seconds=end_time - start_time)}") 