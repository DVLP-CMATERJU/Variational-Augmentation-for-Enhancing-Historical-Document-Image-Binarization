import torch
import config
from torchvision.utils import save_image

def save_some_examples(model, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        y_res = model(x)
        y_res = y_res * 0.5 + 0.5  # remove normalization#
        save_image(y_res, folder + f"/result_{epoch}.jpg")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.jpg")
        save_image(y * 0.5 + 0.5, folder + f"/target_{epoch}.jpg")
    model.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    #for param_group in optimizer.param_groups:
    #    param_group["lr"] = lr