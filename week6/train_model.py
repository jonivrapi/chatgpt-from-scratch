import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt


# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the 
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step. 
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

# ===========================================================================

'''
Complete the following method which trains a GPT model and saves a loss curve.

To reiterate: you don't need to worry about weight decay, weight initialization, grad accumulation, or weight tying.
Use whatever batch size you are able, even something like 2 or 4 is fine.
Use a few hundred warmup steps and a peak learning rate that is (something x 10-4).
'''
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Training on device:", device)

    # adjust as needed
    # here i used the same properties you had in your example
    # except with a larger vocab to fit the tokenizer we trained
    model = GPTModel(d_model=128, n_heads=8, layers=4, vocab_size=10000, max_seq_len=512)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)
    model.train()

    '''
    # TODO
    
    pseudocode:
    opt = torch.optim.AdamW(...
    scheduler = cosine_with_warmup_lr_scheduler(...

    for batch in dataset:
        opt.zero_grad()

        <run batch through model and compute loss>

        # VERY IMPORTANT ---
        torch.nn.CrossEntropyLoss expects classes in the 2nd dimension.
        You may need to transpose dimensions around to make this work.
        It also expects unnormalized logits (no softmax).
        # ---

        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step the optimizer and scheduler
        opt.step()
        scheduler.step() 

        # log total tokens and loss
        # periodically save a plot of loss vs tokens      
        
    '''
    sequences = np.load("dataset.npy", allow_pickle=False)

    inputs = torch.from_numpy(sequences[:, :-1]).long()
    targets = torch.from_numpy(sequences[:, 1:]).long()
    dataset = torch.utils.data.TensorDataset(inputs, targets)

    batch_size = 256
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    total_steps = len(data_loader)
    epochs = 3
    total_steps *= epochs
    warmup_steps = min(500, max(1, total_steps // 10))

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95))
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps)
    criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    tokens_seen = 0
    token_history = []
    loss_history = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    (line,) = ax.plot([], [])
    ax.set_xlabel("Tokens Seen")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve (live)")
    fig.tight_layout()

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            opt.zero_grad()

            logits = model(batch_inputs)
            logits = logits.transpose(1, 2)

            loss = criterion(logits, batch_targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            scheduler.step()

            tokens_seen += batch_targets.numel()
            global_step += 1

            token_history.append(tokens_seen)
            loss_history.append(loss.item())

            if global_step % 100 == 0 or global_step == total_steps:
                print(f"step {global_step}/{total_steps} | tokens {tokens_seen} | loss {loss.item():.4f}")
                line.set_data(token_history, loss_history)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

    plt.ioff()
    fig.savefig("training_loss.png")
    plt.close(fig)

    # save model weights if you want
    torch.save(model.state_dict(), "./model_weights.pt")



if __name__ == "__main__":
    train()
