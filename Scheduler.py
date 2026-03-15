import torch
from tqdm import tqdm

class LinearScheduler:
    def __init__(self , n_ts =1000, beta_s=1e-4 , beta_e=0.02,img_size=128,device=None):
        super().__init__()
        
        self.n_ts = n_ts
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.device = device
        self.img_size=img_size
        
        self.betas = torch.linspace(beta_s , beta_e , n_ts) ## noise scheduler
        
        self.alphas = 1. - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        
        self.alphas_bar  = torch.cumprod(self.alphas , dim=0) ## cumulative product
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        

    def add_noise(self,x0,noise,t,device): ## adding noise to the original image with a given time step 
        x0_shape = x0.shape

        batch_size = x0_shape[0]
        for _ in range(len(x0_shape)-1):
            alphas_bar = self.alphas_bar[t.cpu()].reshape(batch_size,1,1,1)
            sqrt_alphas_bar = self.sqrt_alphas_bar[t.cpu()].reshape(batch_size,1,1,1)
    
        alphas_bar = alphas_bar.to(device)
        sqrt_alphas_bar = sqrt_alphas_bar.to(device)
        return sqrt_alphas_bar*x0 + torch.sqrt(1-alphas_bar)*noise,noise ## starting from x0 and returning xt (Noisy x0 at a given time step t)

    def sample_prev_time_step(self,xt,noise_theta,t):

        x0 = (xt - (torch.sqrt(1-self.alphas_bar[t]))) / self.sqrt_alphas_bar[t] ## from the previous formula in the add_noise function
        x0 = torch.clamp(x0 , min=-1. , max=1)
    
    
        mean_theta = (xt - ((1-self.alphas[t])/torch.sqrt(1 - self.alphas_bar[t])) * noise_theta ) / self.sqrt_alphas[t]
    
        if t==0:
            return mean_theta , x0
        else : 
            sigma = ( ((1-self.alphas[t])*(1-self.alphas_bar[t-1]))/(1-self.alphas_bar[t]) ) ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return (mean_theta + sigma*z) , x0 
    def reverse_diffusion(self,model,n_images,n_channels,pos_embedding_dim,pos_embedding_func,save_time_steps=None,input_image=None):
        with torch.no_grad():
            
            x = torch.randint((n_images,n_channels,self.img_size))
            x = x.to(self.device)
            denoised_images = []
            for i in tqdm(reversed(range(0,self.noise_steps)),desc="U-net inference", total=self.noise_steps):
                t=(torch.ones(n_images)*i).long()
                t_pos_emb = pos_embedding_func(t.unsqueeze(1),pos_embedding_dim).to(self.device)
                predicted_noise = model(torch.cat((input_image.to(self.device),x),dim=1),t_pos_emb)
                alpha = self.alpha[t][:,None,None,None]
                alpha_bar = self.alpha_bar[t][:,None,None,None]
                if i>0 : 
                    noise = torch.randn_like(x)
                else : 
                    noise = torch.zeros_like(x)

                x = (1/torch.sqrt(alpha) * (x-((1-alpha)/torch.sqrt(1-alpha_bar))*predicted_noise)) + torch.sqrt(1-alpha)*noise ## Xt-1
                if i in save_time_steps :
                    denoised_images.append(x)
            denoised_images = torch.stack(denoised_images)
            denoised_images = denoised_images.swapaxes(0,1)
            return denoised_images