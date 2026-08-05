import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.auto import tqdm


class EncodeBetaModel(nn.Module):
    '''
    First convolve the predictor with beta in each event group
    Then convolve the event with the kernel
    Add the beta modification to each event channel
    Sum together
    '''
    def __init__(self, num_events, num_beta, kernel_size, skip_beta=False):
        super().__init__()

        self.conv_beta = nn.Conv1d(in_channels= num_beta*num_events, out_channels = num_events,
                                   groups=num_events,
            kernel_size=kernel_size, padding='same', bias=False)

        self.conv1 = nn.Conv1d(in_channels=num_events, out_channels = num_events, groups = num_events,
                    kernel_size=kernel_size, padding='same', bias=False)
  
    def forward(self, evt, pred):
        signal = self.conv1(evt)
        factor = self.conv_beta(pred)
        output = (factor + signal).sum(axis=0, keepdims=True)
        return output.T
    

def train(model, target, *args, epochs=1000, print_iter=200):
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    pbar = tqdm(range(epochs), unit='Epoch')
    
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(*args)
        l1_penality = torch.norm(model.conv_beta.weight,2)
        loss = criterion(output, target)  #+ 0.001*l1_penality # force beta to be sparse
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch%print_iter == 0:
            pbar.set_description(f'Loss: {loss.item():.4f} lr={optimizer.param_groups[0]["lr"]:.4e}')