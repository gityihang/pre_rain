import torch.nn as nn

#定义分类模型
class ClaModel(nn.Module):  
    def __init__(self, input_size, hidden_size , output_size):  
        super(ClaModel, self).__init__()  
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, output_size)  
        self.relu = nn.ReLU()  
        self.sigmoid = nn.Sigmoid()  
  
    def forward(self, x):  
        x = self.fc1(x)  
        x = self.relu(x)  
        x = self.fc2(x)  
        x = self.sigmoid(x)  
        return x  
    
    
#定义回归模型
class RegModel(nn.Module):
    def __init__(self, input_size, hidden_size , output_size = 1):
        super(RegModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()         

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x