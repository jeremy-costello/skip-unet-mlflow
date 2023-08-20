import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import random
import mlflow
import optuna
import numpy as np
from tqdm import tqdm

from unet_network import UNet


"""
TODO
- ensure no same name between log and uniform keys
- run one epoch of max sizes (list of things that affect memory)
- multiple seeds per hyperparameter configuration
"""

seed = None
num_trials = 100
experiment_name = "skip_unet_optuna"

num_epochs = 20
input_max_dropout = 0.7
input_max_noise_scale = 1.0

tensorboard_dir = "logs/tensorboard"

search_ranges = {
    "log": {
        "learning_rate": (1e-4, 1e-2),
        "1-beta1": (1e-2, 1e-1),
        "1-beta2": (1e-4, 1e-1),
        "epsilon": (1e-10, 1e-6),
        "weight_decay": (1e-3, 1e-1),
        "batch_size": (2**4, 2**6),
        "initial_channels": (2**2, 2**4),
    },
    "uniform": {
        "resize_blocks": (1, 2),
        "resize_layers_per_block": (1, 2),
        "middle_layers": (1, 2),
        "skip_connection": ("addition", "concatenation")
    }
}

# get and set seed
if seed is None:
    seed = torch.randint(2 ** 32, (1, 1)).item()

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

random.seed(seed)
np.random.seed(seed)

# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {device}")

# load dataset
dataset = datasets.MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = dataset(root='./data', train=True, download=True, transform=transform)
test_dataset = dataset(root='./data', train=False, download=True, transform=transform)



def main():   
    study = optuna.create_study(direction="minimize", study_name=experiment_name)
    study.optimize(objective, n_trials=num_trials)

    with mlflow.start_run(run_name="common"):
        # log constant params
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("input_max_dropout", input_max_dropout)
        mlflow.log_param("input_max_noise_scale", input_max_noise_scale)
        mlflow.log_param("seed", seed)

        # log dataset
        mlflow.log_artifact(f"./data/{dataset.__name__}", artifact_path="datasets")

        # log final params
        mlflow.log_params(study.best_params)
        mlflow.log_metric("BEST_VALIDATION_LOSS", study.best_value)


def objective(trial):
    hyperparameter_dict = dict()

    for scale_type, range_dict in search_ranges.items():
        assert scale_type in ["log", "uniform"]
        to_log = scale_type == "log"
        for hyp_name in range_dict.keys():
            if isinstance(range_dict[hyp_name][0], str):
                assert len(range_dict[hyp_name]) > 1
                for hyp in range_dict[hyp_name]:
                    assert isinstance(hyp, str)
            elif len(range_dict[hyp_name]) == 2:
                assert type(range_dict[hyp_name][0]) == type(range_dict[hyp_name][1])
                assert isinstance(range_dict[hyp_name][0], float) or isinstance(range_dict[hyp_name][0], int)
            else:
                raise ValueError(f"Invalid hyperparameter range in {hyp_name}.")

            if isinstance(range_dict[hyp_name][0], float):
                hyperparameter_dict[hyp_name] = trial.suggest_float(hyp_name, range_dict[hyp_name][0],
                                                                    range_dict[hyp_name][1], log=to_log)
            elif isinstance(range_dict[hyp_name][0], int):
                hyperparameter_dict[hyp_name] = trial.suggest_int(hyp_name, range_dict[hyp_name][0],
                                                                range_dict[hyp_name][1], log=to_log)
            elif isinstance(range_dict[hyp_name][0], str):
                hyperparameter_dict[hyp_name] = trial.suggest_categorical(hyp_name, list(range_dict[hyp_name]))
            else:
                raise ValueError(f"Invalid hyperparameter type in {hyp_name}")
    
    # build some hyperparameters
    batch_size = hyperparameter_dict["batch_size"]
    print(f"BATCH SIZE: {batch_size}")
    resize_blocks = [hyperparameter_dict["resize_layers_per_block"]] * hyperparameter_dict["resize_blocks"]
    betas = (1 - hyperparameter_dict["1-beta1"], 1 - hyperparameter_dict["1-beta2"])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_examples = len(train_loader)

    for batch in train_loader:
        image_channels = batch[0].size(1)
        break

    
    arch_dict = {
        'image_channels': image_channels,
        'initial_channels': hyperparameter_dict["initial_channels"],
        'resize_blocks': resize_blocks,
        'middle_blocks': hyperparameter_dict["middle_layers"],
        'skip_connection': hyperparameter_dict["skip_connection"]
    }

    # Define your PyTorch modules
    unet_net = UNet(arch_dict)
    unet_net.to(device)

    total_params = sum(param.numel() for param in unet_net.parameters())
    print(f"TOTAL PARAMETERS: {total_params}")

    # Define the optimizer
    optimizer = optim.AdamW(unet_net.parameters(), lr=hyperparameter_dict["learning_rate"], betas=betas,
                            eps=hyperparameter_dict["epsilon"], weight_decay=hyperparameter_dict["weight_decay"])

    with SummaryWriter(tensorboard_dir) as writer:
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            # log optimizer parameters
            for i, group in enumerate(optimizer.param_groups):
                for key in group.keys():
                    if key == "params":
                        continue
                    else:
                        mlflow.log_param(f"g{i}_{key}", group[key])
            
            # log model parameters
            for key in arch_dict.keys():
                mlflow.log_param(key, arch_dict[key])
            
            # log other parameters
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("total_parameters", total_params)

            best_avg_val_loss = float('inf')
            for epoch in range(num_epochs):
                unet_net.train()

                losses = []
                with tqdm(train_loader, leave=False, ncols=80) as pbar:

                    for step, (images, labels) in enumerate(pbar):
                        images = images.to(device)

                        # Zero the gradients
                        optimizer.zero_grad()
                        
                        bsz = images.size(0)
                        channels = images.size(1)
                        scale = input_max_noise_scale * torch.rand((bsz, channels, 1, 1)).to(device)
                        input = images + scale * torch.randn_like(images)
                        input = torch.clip(input, min=-1.0, max=1.0)
                        input_dropout = torch.rand(1).item() * input_max_dropout
                        input = F.dropout(input, p=input_dropout)

                        # Forward pass
                        output = unet_net(input)

                        # Compute the loss
                        loss = F.mse_loss(output, images)
                        losses.append(loss)

                        # Backward pass and optimization
                        loss.backward()
                        optimizer.step()

                        # log training loss for each step
                        mlflow.log_metric("train_loss_step", loss.item(), step=step + epoch * train_examples)
                        writer.add_scalar("train_loss_step", loss.item(), step + epoch * train_examples)

                        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
                        pbar.update()
                
                # Compute validation loss
                unet_net.eval()  # Switch to evaluation mode
                val_losses = []
                with torch.no_grad():
                    for val_images, val_labels in test_loader:
                        val_images = val_images.to(device)

                        bsz = val_images.size(0)
                        channels = val_images.size(1)
                        scale = input_max_noise_scale * torch.rand((bsz, channels, 1, 1)).to(device)

                        val_input = val_images + scale * torch.randn_like(val_images)
                        val_input = torch.clip(val_input, min=-1.0, max=1.0)
                        val_input_dropout = torch.rand(1).item() * input_max_dropout
                        val_input = F.dropout(val_input, p=val_input_dropout)

                        # Forward pass
                        val_output = unet_net(val_input)

                        # Compute the loss
                        val_loss = F.mse_loss(val_output, val_images)
                        val_losses.append(val_loss)

                # Calculate average validation loss
                avg_val_loss = torch.mean(torch.stack(val_losses)).item()
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    torch.save(unet_net.state_dict(), "model.ckpt")

                # log avergae validation loss for each epoch
                mlflow.log_metric("avg_val_loss_epoch", avg_val_loss, step=epoch)
                writer.add_scalar("avg_val_loss_epoch", avg_val_loss, epoch)

                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            
            # final mlflow logging
            mlflow.pytorch.log_model(unet_net, "models")
            mlflow.log_artifact("model.ckpt", artifact_path="checkpoints")
            mlflow.log_artifact(tensorboard_dir, artifact_path="tensorboard_logs")
            mlflow.log_metric("best_avg_val_loss", best_avg_val_loss)
    
    return best_avg_val_loss


if __name__ == "__main__":
    main()
        