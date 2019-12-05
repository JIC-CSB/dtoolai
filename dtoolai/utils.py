

def train(model, dl, optimiser, loss_fn, n_epochs):
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
