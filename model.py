import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


"""
This file contains the model classes for the different models used in the project.
These models were developed iteratively as the project progressed, starting with the simplest. 
"""

class SimpleLSTM(torch.nn.Module):
    """
    Simple LSTM model 
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out
    


class ExtendedLSTM(nn.Module):
    """
    Extended LSTM model with additional fully connected layers for scalar prediction
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, scalar_output_dim, ann_hidden_dims):
        super(ExtendedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc_next_step = nn.Linear(hidden_dim, output_dim)

        self.ann_layers = nn.ModuleList()
        last_dim = hidden_dim  
        for next_dim in ann_hidden_dims:
            self.ann_layers.append(nn.Linear(last_dim, next_dim))
            self.ann_layers.append(nn.ReLU())
            last_dim = next_dim

        self.fc_scalar_final = nn.Linear(last_dim, scalar_output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        next_step_prediction = self.fc_next_step(lstm_out[:, -1, :])

        ann_out = hn[-1]  
        for layer in self.ann_layers:
            ann_out = layer(ann_out)

        scalar_prediction = self.fc_scalar_final(ann_out)

        return next_step_prediction, scalar_prediction

# Did not work well 
class Autoencoder(nn.Module):
    """
    Autoencoder model for dimensionality reduction of brain region 
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(86, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 86),  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# FINAL MODEL 
class LSTMwithStatic(nn.Module):
    """
    LSTM model with additional fully connected layers for scalar prediction and static demographic inputs 
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, scalar_output_dim, ann_hidden_dims, static_input_dim):
        """
        Initialize the model

        Params
        ------
        input_dim: int
            Dimension of input data
        hidden_dim: int
            Dimension of hidden layer in LSTM
        layer_dim: int
            Number of layers in LSTM
        output_dim: int
            Dimension of output data   
        scalar_output_dim: int
            Dimension of scalar output data
        ann_hidden_dims: list
            List of hidden layer dimensions for fully connected layers
        static_input_dim: int
            Dimension of static input data

        """
        super(LSTMwithStatic, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.total_input_dim = input_dim + static_input_dim
        self.lstm = nn.LSTM(self.total_input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc_next_step = nn.Linear(hidden_dim, output_dim)  
        self.ann_layers = nn.ModuleList()
        last_dim = hidden_dim  
        
        # Fully connected layers for scalar prediction
        for next_dim in ann_hidden_dims:
            self.ann_layers.append(nn.Linear(last_dim, next_dim))
            self.ann_layers.append(nn.ReLU())
            last_dim = next_dim

        self.fc_scalar_final = nn.Linear(last_dim + static_input_dim, scalar_output_dim)

    def forward(self, dynamic_input, static_input):
        """
        Forward pass through the model

        Params
        ------
        dynamic_input: torch.Tensor
            Dynamic input data (i.e. timeseries)
        static_input: torch.Tensor
            Static input data (i.e. demographic information)

        Returns
        -------
        next_step_prediction: torch.Tensor
            Prediction for the next timestep
        scalar_prediction: torch.Tensor
            Prediction for the scalar output

        """
        static_input_repeated = static_input.unsqueeze(1).repeat(1, dynamic_input.size(1), 1)
        combined_input = torch.cat((dynamic_input, static_input_repeated), dim=2)

        h0 = torch.zeros(self.layer_dim, combined_input.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, combined_input.size(0), self.hidden_dim).requires_grad_()

        lstm_out, (hn, cn) = self.lstm(combined_input, (h0.detach(), c0.detach()))

        next_step_prediction = self.fc_next_step(lstm_out[:, -1, :])

        ann_out = hn[-1]
        for layer in self.ann_layers:
            ann_out = layer(ann_out)
        
        ann_out_combined = torch.cat((ann_out, static_input), dim=1)
        scalar_prediction = self.fc_scalar_final(ann_out_combined)

        return next_step_prediction, scalar_prediction
    

# Benchmarking model 
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        last_dim = input_dim
        for next_dim in hidden_dims:
            self.layers.append(nn.Linear(last_dim, next_dim))
            self.layers.append(nn.ReLU())
            last_dim = next_dim

        self.layers.append(nn.Linear(last_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------------------------
# Model that includes classification for diagnosis, doesn't do diagnosis well! 
class AlzNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, scalar_output_dim, ann_hidden_dims, static_input_dim, num_classes):
        super(AlzNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.total_input_dim = input_dim + static_input_dim
        self.lstm = nn.LSTM(self.total_input_dim, hidden_dim, layer_dim, batch_first=True)

        # Existing layers
        self.fc_next_step = nn.Linear(hidden_dim, output_dim)
        self.ann_layers = nn.ModuleList()
        last_dim = hidden_dim

        for next_dim in ann_hidden_dims:
            self.ann_layers.append(nn.Linear(last_dim, next_dim))
            self.ann_layers.append(nn.ReLU())
            last_dim = next_dim

        self.fc_scalar_final = nn.Linear(last_dim + static_input_dim, scalar_output_dim)

        # classification network 
        for next_dim in ann_hidden_dims:
            self.ann_layers.append(nn.Linear(last_dim, next_dim))
            self.ann_layers.append(nn.ReLU())
            last_dim = next_dim

        self.fc_classification = nn.Linear(last_dim, num_classes) 

    def forward(self, dynamic_input, static_input, training=False):
        static_input_repeated = static_input.unsqueeze(1).repeat(1, dynamic_input.size(1), 1)
        combined_input = torch.cat((dynamic_input, static_input_repeated), dim=2)

        h0 = torch.zeros(self.layer_dim, combined_input.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, combined_input.size(0), self.hidden_dim).requires_grad_()

        lstm_out, (hn, cn) = self.lstm(combined_input, (h0.detach(), c0.detach()))

        next_step_prediction = self.fc_next_step(lstm_out[:, -1, :])

        ann_out = hn[-1]
        for layer in self.ann_layers:
            ann_out = layer(ann_out)

        ann_out_combined = torch.cat((ann_out, static_input), dim=1)
        scalar_prediction = self.fc_scalar_final(ann_out_combined)

        classification = hn[-1]
        for layer in self.ann_layers:
            classification = layer(classification)


        classification_output = self.fc_classification(classification_output)  

        if training:
            return next_step_prediction, scalar_prediction, classification_output
        
        classification_probs = torch.softmax(classification_output, dim=1)  
        return next_step_prediction, scalar_prediction, classification_probs



    

