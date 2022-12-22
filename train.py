import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import BinDataset, SynDataset
from model import build_res_unet
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from synthetic.test import synthesize
from torchvision.utils import save_image

#torch.backends.cudnn.benchmark = True
def pretrain_generator(net_G, train_dl, l1_loss, opt, epochs):
    for ep in range(epochs):
        for x, y in tqdm(train_dl):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            preds = net_G(x)
            loss = l1_loss(preds, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        print(f"Epoch {ep + 1}/{epochs}")
        print(f"L1 Loss: {loss.mean().item():.5f}")
        save_checkpoint(net_G, opt, filename=config.CHECKPOINT_GEN)


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (src, tgt) in enumerate(loop):
        src = src.to(config.DEVICE)
        tgt = tgt.to(config.DEVICE)
        #print(src.shape, tgt.shape)
        src, tgt = synthesize(src, tgt)
        # Train Discriminator
        with torch.cuda.amp.autocast():
            tgt_fake = gen(src)
            D_real = disc(src, tgt)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(src, tgt_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(src, tgt_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(tgt_fake, tgt) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                real=torch.sigmoid(D_real).mean().item(),
                fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = build_res_unet(config.DEVICE, n_input=3, n_output=1, size=256)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    #train_dataset = SynDataset(source_dir=config.TRAIN_DIR, transform=config.dibco_transforms)

    
    train_dataset = BinDataset(source_dir=config.TRAIN_DIR+"/cl_patches", 
                        target_dir=config.TRAIN_DIR+"/gt_patches",
                        transform=config.dibco_transforms
                        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #val_dataset = SynDataset(source_dir=config.VAL_DIR, transform=config.dibco_transforms)
    
    val_dataset = BinDataset(source_dir=config.VAL_DIR+"/cl_patches", 
                        target_dir=config.VAL_DIR+"/gt_patches",
                        transform=config.dibco_transforms
                        )
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)        
    
    print("pretraining genrator:")
    pretrain_generator(gen, train_loader, L1_LOSS, opt_gen, config.PREPOCHS)

    if config.LOAD_GMODEL:
        gen = nn.DataParallel(gen)
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
    if config.LOAD_DMODEL:
        disc = nn.DataParallel(disc)
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )
    
    print("starting GAN training:")
    gen = nn.DataParallel(gen)
    disc = nn.DataParallel(disc)
    for epoch in range(config.NUM_EPOCHS):
        print(f"{epoch}/{config.NUM_EPOCHS}")
        
        gen.train()
        disc.train()
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and epoch % 1 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

            save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()