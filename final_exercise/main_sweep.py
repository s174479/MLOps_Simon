### Something not working!!!

import argparse
import sys

import torch
import torchvision.transforms as transforms
import click

from data import mnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt

import wandb # Added for logging S4 M13

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # Define sweep config
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.0001}
        }
    }

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='MNIST-first-sweep')
    
    # TODO: Implement training loop here
    def main():
        run = wandb.init()

        # note that we define values from `wandb.config` instead 
        # of defining hard values
        lr  =  wandb.config.lr
        epochs = wandb.config.epochs

        model = MyAwesomeModel()
        trainloader, _ = mnist()
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        steps = 0

        train_losses = []
        for e in range(epochs):
            print("Epoch:", e)
            running_loss = 0
            for images, labels in trainloader:
                
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            train_losses.append(running_loss)
            wandb.log({'epoch': e, 'train_loss': running_loss})
            
        torch.save(model.state_dict(), 'checkpoint.pth')
        plt.plot(range(epochs), train_losses)
        plt.xlabel("Training step (epoch)")
        plt.ylabel("Training loss")
        plt.show()

    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=4)


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    _, test_set = mnist()
    ## TODO: Implement the validation pass and print out the validation accuracy
    with torch.no_grad():
        # Validation pass
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    