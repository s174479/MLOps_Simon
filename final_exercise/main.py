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

    wandb.init(project="MNIST project", entity="s174479_mlops") # Added for logging S4 M13
    
    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    #trainloader, _ = mnist()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 5
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
            wandb.log({"loss": loss}) # Added for logging S4 M13

        train_losses.append(running_loss)
        

    torch.save(model.state_dict(), 'checkpoint.pth')
    plt.plot(range(epochs), train_losses)
    plt.xlabel("Training step (epoch)")
    plt.ylabel("Training loss")
    wandb.log({"Running loss pr. epoch": plt}) # Added for logging S4 M13
    plt.show()


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
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    ## TODO: Implement the validation pass and print out the validation accuracy
    with torch.no_grad():
        # Validation pass
        for images, labels in testloader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    