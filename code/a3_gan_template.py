import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(in_features=args.latent_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(128),

            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),

            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(512),

            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(in_features=1024, out_features=784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.layers(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    criterion = nn.BCELoss()
    for epoch in range(args.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):
            # flatten images
            imgs = imgs.view(-1, 784).cuda()

            batch_size = len(imgs)
            latent_dim = args.latent_dim

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()

            # get noisy latent space vector
            noisy_vectors = torch.randn((batch_size, latent_dim)).to(device)

            # generates images from noisy latent space vector
            gen_imgs = generator(noisy_vectors)

            # labels generated images as real
            generated_lbs = torch.ones(gen_imgs.shape[0]).to(device)

            # computes the loss as the fake images are not intrepreted as real
            generator_loss = criterion(discriminator(gen_imgs).squeeze(), generated_lbs)
            generator_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            real_lbs = torch.ones(imgs.shape[0]).to(device)  # classifying images as real
            generated_lbs = generated_lbs - 1  # label generated images as fake

            # computes the loss taking in account correct classified real and fake images
            discriminator_loss = (criterion(discriminator(imgs).squeeze(), real_lbs) + criterion(
                discriminator(gen_imgs.detach()).squeeze(),
                generated_lbs)) / 2

            gen_imgs = gen_imgs.detach()

            discriminator_loss.backward()

            optimizer_D.step()

            #

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_imgs[:25].view(-1, 1, 28, 28),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    device = torch.device("cuda")

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
