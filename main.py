import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

class HangulDataset(Dataset):
    def __init__(self, image_paths, font_labels, transform=None):
        self.image_paths = image_paths
        self.font_labels = font_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        font_label = self.font_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, font_label

class Encoder(nn.Module):
    def __init__(self, input_dim=64*64, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, font_dim=10):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.font_dim = font_dim
        
        self.fc = nn.Linear(latent_dim + font_dim, 256 * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        
    def forward(self, z, font_vector):
        combined = torch.cat([z, font_vector], dim=1)
        x = F.relu(self.fc(combined))
        x = x.view(x.size(0), 256, 4, 4)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim=128, font_dim=10):
        super(Generator, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, font_dim=font_dim)
        
    def forward(self, source_image, font_vector):
        z = self.encoder(source_image)
        generated_image = self.decoder(z, font_vector)
        return generated_image

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        # 판별자의 첫 레이어에는 보통 배치 정규화를 적용하지 않습니다.
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc = nn.Linear(256 * 4 * 4, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        
        return x

class HandwritingCorrectionSystem:
    def __init__(self, latent_dim=128, num_fonts=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_fonts = num_fonts
        self.latent_dim = latent_dim
        self.font_dim = num_fonts
        
        self.generator = Generator(self.latent_dim, self.font_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        self.font_embeddings = nn.Embedding(self.num_fonts, self.font_dim).to(device)
        
    def create_font_vector(self, font_id, batch_size):
        font_ids = torch.full((batch_size,), font_id, dtype=torch.long).to(self.device)
        return self.font_embeddings(font_ids)
    
    def interpolate_font_vectors(self, font_id1, font_id2, num_steps=3):
        font_vec1 = self.font_embeddings(torch.tensor([font_id1]).to(self.device))
        font_vec2 = self.font_embeddings(torch.tensor([font_id2]).to(self.device))
        
        interpolated_vectors = []
        for i in range(num_steps + 2):
            alpha = i / (num_steps + 1)
            interpolated = (1 - alpha) * font_vec1 + alpha * font_vec2
            interpolated_vectors.append(interpolated)
        
        return interpolated_vectors
    
    def train_step(self, real_images, source_images, target_font_ids):
        batch_size = real_images.size(0)
        
        # One-sided Label Smoothing: 진짜 레이블을 1.0 대신 0.9로 설정
        real_labels = torch.full((batch_size, 1), 0.9, dtype=torch.float, device=self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        font_vectors = []
        for font_id in target_font_ids:
            font_vec = self.create_font_vector(font_id.item(), 1)
            font_vectors.append(font_vec)
        font_vectors = torch.cat(font_vectors, dim=0)
        
        self.d_optimizer.zero_grad()
        
        real_pred = self.discriminator(real_images)
        d_real_loss = self.adversarial_loss(real_pred, real_labels)
        
        fake_images = self.generator(source_images, font_vectors)
        fake_pred = self.discriminator(fake_images.detach())
        d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()
        
        self.g_optimizer.zero_grad()
        
        fake_pred = self.discriminator(fake_images)
        g_adv_loss = self.adversarial_loss(fake_pred, real_labels)
        
        g_l1_loss = self.l1_loss(fake_images, real_images)
        
        g_loss = g_adv_loss + 100 * g_l1_loss
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_l1_loss': g_l1_loss.item()
        }
    
    def generate_correction_guideline(self, user_handwriting, target_font_id, num_steps=3):
        self.generator.eval()
        
        with torch.no_grad():
            user_font_id = 0
            
            interpolated_vectors = self.interpolate_font_vectors(
                user_font_id, target_font_id, num_steps
            )
            
            correction_steps = []
            for font_vector in interpolated_vectors:
                batch_size = user_handwriting.size(0)
                font_vector_expanded = font_vector.expand(batch_size, -1)
                
                generated_image = self.generator(user_handwriting, font_vector_expanded)
                correction_steps.append(generated_image)
        
        return correction_steps
    
    def save_models(self, filepath):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'font_embeddings_state_dict': self.font_embeddings.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }, filepath)
    
    def load_models(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.font_embeddings.load_state_dict(checkpoint['font_embeddings_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])


def load_hangul_data(data_dir):
    image_paths = glob.glob(os.path.join(data_dir, '*', '*.jpeg'))
    
    font_names = sorted(list(set([os.path.basename(p).split('_')[0] for p in image_paths])))
    font_map = {name: i for i, name in enumerate(font_names)}
    
    font_labels = [font_map[os.path.basename(p).split('_')[0]] for p in image_paths]
    
    print(f"총 {len(image_paths)}개의 이미지를 찾았습니다.")
    print(f"총 {len(font_names)}개의 폰트가 있습니다.")
    
    return image_paths, font_labels, len(font_names)

def visualize_correction_process(correction_steps, save_path=None):
    num_steps = len(correction_steps)
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
    
    for i, step_image in enumerate(correction_steps):
        img = step_image[0].cpu().numpy().squeeze()
        img = (img + 1) / 2
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Step {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    data_dir = './hangul_images'
    epochs = 25
    batch_size = 128
    
    image_paths, font_labels, num_fonts = load_hangul_data(data_dir)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = HangulDataset(image_paths=image_paths, font_labels=font_labels, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    system = HandwritingCorrectionSystem(
        num_fonts=num_fonts,
        latent_dim=128, # latent_dim도 명시적으로 전달
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("훈련을 시작합니다...")
    for epoch in range(epochs):
        for i, (images, font_ids) in enumerate(data_loader):
            
            real_images = images.to(system.device)
            target_font_ids = font_ids.to(system.device)
            
            source_images = real_images
            
            losses = system.train_step(real_images, source_images, target_font_ids)
            
            if (i + 1) % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(data_loader)}] "
                      f"[D loss: {losses['d_loss']:.4f}] [G loss: {losses['g_loss']:.4f}] "
                      f"[G adv: {losses['g_adv_loss']:.4f}, G L1: {losses['g_l1_loss']:.4f}]")

        system.save_models(f"hangul_font_model_epoch_{epoch+1}.pth")

    print("훈련 완료!")
    
    print("\n훈련된 모델로 교정 가이드라인 생성 테스트:")
    system.generator.eval()
    
    test_img, test_font_id = next(iter(data_loader))
    user_handwriting = test_img[:1].to(system.device)
    
    target_font_id = 5
    if num_fonts <= target_font_id:
        target_font_id = num_fonts - 1

    correction_steps = system.generate_correction_guideline(
        user_handwriting, 
        target_font_id=target_font_id,
        num_steps=4
    )
    
    print(f"생성된 교정 단계 수: {len(correction_steps)}")
    visualize_correction_process(correction_steps, save_path="correction_guideline.png")

if __name__ == "__main__":
    main()