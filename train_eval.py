import torch
from utils import accuracy


def train_epoch(model, train_dataloader, optimizer, loss_fn, device):
    train_loss = 0
    model.train()
    model.to(device)
    for words, indices in train_dataloader:
        words.to(device)
        indices.to(device)

        optimizer.zero_grad()
        # Forward pass
        indices_output = model(words.to(device))

        # Calculate loss
        loss = loss_fn(indices_output, indices)
        train_loss += loss

        # Combine the losses
        loss.backward()

        optimizer.step()

    train_loss /= len(train_dataloader)
    return train_loss


def eval_epoch(model, test_dataloader, loss_fn, device):

    test_loss = 0
    test_accuracy = 0
    with torch.inference_mode():
        model.to(device)
        for words, indices in test_dataloader:
            indices.to(device)
            words.to(device)
            # Forward pass
            indices_output = model(words)

            # Calculate loss
            test_loss += loss_fn(indices_output, indices)
            test_accuracy += accuracy(indices, indices_output.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

    return test_loss, test_accuracy
