import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# define the hyperparameters
num_epochs = 50
learning_rate = 2e-4

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        # 14 x 14 x 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        # 7 x 7 x 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        # 3 x 3 x 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False)
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.sigmoid(self.layer4(out)).squeeze()
        return out
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128),
            nn.SELU()
        )
        # 4 x 4 x 128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SELU()
        )
        # 8 x 8 x 64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SELU()
        )
        # 16 x 16 x 32
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=3, bias=False),
            nn.Tanh()
        )
        # 28 x 28 x 1
        
        
    def forward(self, x):
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out
        
generator = Generator()
discriminator = Discriminator()

loss = nn.BCELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)



def generate_noise():
    minibatch_size = 100
    noise = Variable(torch.from_numpy(
        np.random.randn(minibatch_size, 128, 1, 1).astype(np.float32)))
    return noise

def update_discriminator(inputs, zeros, ones):
    # Zero gradients for the discriminator
    disc_optimizer.zero_grad()

    # Train with real examples
    d_real = discriminator(inputs)

    d_real_loss = loss(d_real, ones)  # Train discriminator to recognize real examples
    d_real_loss.backward()

    # Train with fake examples from the generator
    fake = generator.forward(generate_noise()).detach()  # Detach to prevent backpropping through the generator
    d_fake = discriminator(fake)

    d_fake_loss = loss(d_fake, zeros)  # Train discriminator to recognize generator samples
    d_fake_loss.backward()
    minibatch_disc_losses.append(d_real_loss.data[0] + d_fake_loss.data[0])

    # Update the discriminator
    disc_optimizer.step()
    
def update_generator(inputs, ones):
    # Zero gradients for the generator
    gen_optimizer.zero_grad()

    d_fake = discriminator(generator(generate_noise()))
    g_loss = loss(d_fake, ones)  # Train generator to fool the discriminator into thinking these are real.

    g_loss.backward()

    # Update the generator
    gen_optimizer.step()

    minibatch_gen_losses.append(g_loss.data[0])
    
minibatch_disc_losses = []
minibatch_gen_losses = []

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = Variable(inputs)
        labels = Variable(labels)
        
        # truth values for the discriminator imputs
        zeros = Variable(torch.zeros(inputs.size(0)))
        ones = Variable(torch.ones(inputs.size(0)))
        
        update_discriminator(inputs, zeros, ones)
        update_generator(inputs, ones)
    
    print('Generator loss : %.3f' % (np.mean(minibatch_gen_losses)))
    print('Discriminator loss : %.3f' % (np.mean(minibatch_disc_losses)))