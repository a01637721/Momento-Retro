"""
Modelo Baseline: LSTM simple para reconocimiento de acciones
"""
import torch
import torch.nn as nn
import config


class BaselineLSTM(nn.Module):
    """Modelo baseline simple con LSTM"""
    
    def __init__(self, input_size, num_classes, hidden_size=None, num_layers=None, dropout=None):
        super(BaselineLSTM, self).__init__()
        
        # Usar configuración por defecto si no se especifica
        hidden_size = hidden_size or config.BASELINE_CONFIG['hidden_size']
        num_layers = num_layers or config.BASELINE_CONFIG['num_layers']
        dropout = dropout if dropout is not None else config.BASELINE_CONFIG['dropout']
        bidirectional = config.BASELINE_CONFIG['bidirectional']
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Capa de salida
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            logits: [batch_size, num_classes]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Usar el último output de la secuencia
        # lstm_out: [batch_size, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Clasificación
        logits = self.fc(last_output)
        
        return logits

