import torch

import numpy as np

from imageio import imsave

def evaluate_model(model, dl_eval):

    model.eval()
    correct = 0
    n_labels = 0
    with torch.no_grad():
        for data, label in dl_eval:
            n_labels += len(label)
            Y_pred = model(data)
            pred = Y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

        print(f"{correct}/{n_labels}")


def evaluate_model_verbose(model, dl_eval):

    model.eval()
    correct = 0
    n_labels = 0
    with torch.no_grad():
        for data, label in dl_eval:

            model_input = data
            print(model_input)
            model_input_np = model_input.squeeze().cpu().numpy()
            model_input_t = np.transpose(model_input_np, (1, 2, 0))
            imsave('scratch/model_input_t.png', model_input_t)

            print(model_input.shape, model_input.min(), model_input.max(), model_input.dtype)
            n_labels += len(label)
            Y_pred = model(data)
            print(Y_pred)
            pred = Y_pred.argmax(dim=1, keepdim=True)
            print(pred, label)
            correct += pred.eq(label.view_as(pred)).sum().item()

        print(f"{correct}/{n_labels}")


def train(model, dl, optimiser, loss_fn, n_epochs, dl_eval=None):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for n, (data, label) in enumerate(dl):
            optimiser.zero_grad()
            Y_pred = model(data)

            loss = loss_fn(Y_pred, label)
            loss.backward()
            epoch_loss += loss.item()

            optimiser.step()

            if n % 10 == 0:
                print(f"  Epoch {epoch}, batch {n}/{len(dl)}, running loss {epoch_loss}")

        print("Epoch training loss", epoch_loss)
        if dl_eval:
            evaluate_model(model, dl_eval)
