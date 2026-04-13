import numpy as np
import torch
import wandb

from configs.config import MODEL_NAME, EPOCHS, BATCH_SIZE, LR, LOSS, TRAIN_SIZE, VAL_SIZE, SAVE_PATH


def train(train_loader, val_loader, device, wandb_entity):
    # training loop
    # load model
    model = torch.hub.load("intel-isl/MiDaS", MODEL_NAME)
    model.to(device)

    criterion = torch.nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # init wandb
    wandb.init(
        entity=wandb_entity,
        project="depth_benchmarking",
        name=f"{MODEL_NAME}-finetuned-kitti",
        config={
            "model": MODEL_NAME,
            "dataset": "KITTI Stereo 2015",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "loss": LOSS,
            "train_size": TRAIN_SIZE,
            "val_size": VAL_SIZE
        }
    )

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for imgs, depths in train_loader:
            imgs = imgs.to(device)
            depths = depths.to(device)

            pred = model(imgs)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred, size=depths.shape[1:],
                mode='bicubic', align_corners=False
            ).squeeze(1)

            mask = depths > 0
            if mask.sum() == 0:
                continue

            loss = criterion(pred[mask], depths[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_absrels = []

        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs = imgs.to(device)
                depths = depths.to(device)

                pred = model(imgs)
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                pred = torch.nn.functional.interpolate(
                    pred, size=depths.shape[1:],
                    mode='bicubic', align_corners=False
                ).squeeze(1)

                mask = depths > 0
                if mask.sum() == 0:
                    continue

                loss = criterion(pred[mask], depths[mask])
                val_losses.append(loss.item())

                absrel = torch.mean(torch.abs(pred[mask] - depths[mask]) / depths[mask])
                val_absrels.append(absrel.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_absrel = np.mean(val_absrels)

        print(f"epoch {epoch+1}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f} | absrel={val_absrel:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_absrel": val_absrel,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → saved best model at epoch {epoch+1}")

    print(f"done. best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")
    wandb.log({"best_val_loss": best_val_loss, "best_epoch": best_epoch})
    wandb.finish()
