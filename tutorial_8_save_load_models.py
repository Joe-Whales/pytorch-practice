import torch
import torch.nn as nn

# # Example model
# model = nn.Linear(1, 1)
# PATH = ''

# # Not recommended, requires exact file structure etc 
# #Save 
# torch.save(model.state_dict(), PATH)
# # The serialized data is bound to the specific classes and exact directory structure used when the model is saved.
# #Load
# model = torch.load(PATH)
# model.eval()

# # Recommended
# #Save
# torch.save(model.state_dict(), PATH)

# #Load
# #model = Model(*args, **kwargs) - create model instance
# model = nn.Linear(1, 1)
# model.load_state_dict(torch.load(PATH))  # takes in the loaded dictionary
# model.eval()

class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Lazy
# #model = Model(input_size=6)

# FILE = 'model.pth'
# #torch.save(model, FILE)

# model = torch.load(FILE)
# model.eval()

# for param in model.parameters():
#     print(param)

# Prefered Method
model = Model(input_size=6)
FILE = 'model.pth'
torch.save(model.state_dict(), FILE)

for param in model.parameters():
    print(param)

loaded_model = Model(input_size=6)
loaded_model.load_state_dict(torch.load(FILE))  
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)

# Checkpoints
model = Model(input_size=6)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())
checkpoint = {
    'epoch': 90,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}

torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
loaded_checkpoint = torch.load('checkpoint.pth')
epoch = loaded_checkpoint['epoch']
model = Model(input_size=6)
model.load_state_dict(checkpoint['model_state'])
optimizer = torch.optim.SGD(model.parameters(), lr=0)  # lr=0 to not change learning rate
optimizer.load_state_dict(checkpoint['optim_state'])
print(optimizer.state_dict())

# When using a GPU (Save on GPU, Load on CPU)
device = torch.device('cuda')
model = Model(input_size=6).to(device)
torch.save(model.state_dict(), FILE)

device = torch.device('cpu')
model = Model(input_size=6)
model.load_state_dict(torch.load(FILE, map_location=device))

# Both on GPU
device = torch.device('cuda')
model = Model(input_size=6).to(device)
torch.save(model.state_dict(), FILE)

model = Model(input_size=6)
model.load_state_dict(torch.load(FILE))
model.to(device)

# Save on CPU, Load on GPU
model = Model(input_size=6)
torch.save(model.state_dict(), FILE)

device = torch.device('cuda')
model = Model(input_size=6)
model.load_state_dict(torch.load(FILE), map_location='cuda:0') # choose the GPU device
model.to(device)
