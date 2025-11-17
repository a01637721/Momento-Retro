"""
Modelo Mejorado: LSTM bidireccional con atención y regularización
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Attention(nn.Module):
    """Módulo de atención para LSTM"""
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: [batch_size, seq_len, hidden_size]
        Returns:
            context: [batch_size, hidden_size]
            attention_weights: [batch_size, seq_len]
        """
        # Calcular scores de atención
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Softmax para obtener pesos
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Aplicar pesos
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  # [batch_size, hidden_size]
        
        return context, attention_weights


class ImprovedLSTM(nn.Module):
    """Modelo mejorado con LSTM bidireccional, atención y regularización"""
    
    def __init__(self, input_size, num_classes, hidden_size=None, num_layers=None, dropout=None):
        super(ImprovedLSTM, self).__init__()
        
        # Usar configuración por defecto si no se especifica
        hidden_size = hidden_size or config.IMPROVED_CONFIG['hidden_size']
        num_layers = num_layers or config.IMPROVED_CONFIG['num_layers']
        dropout = dropout if dropout is not None else config.IMPROVED_CONFIG['dropout']
        bidirectional = config.IMPROVED_CONFIG['bidirectional']
        use_batch_norm = config.IMPROVED_CONFIG['use_batch_norm']
        use_attention = config.IMPROVED_CONFIG['use_attention']
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.use_attention = use_attention
        
        # Capa de entrada con batch norm opcional
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Tamaño de salida del LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Módulo de atención
        if use_attention:
            self.attention = Attention(lstm_output_size)
        
        # Capas fully connected con dropout
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_output_size // 2, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            logits: [batch_size, num_classes]
        """
        # Batch normalization en la entrada (aplicar por feature)
        if self.use_batch_norm:
            # Transponer para batch norm: [batch_size, input_size, seq_len]
            x = x.transpose(1, 2)
            x = self.input_bn(x)
            x = x.transpose(1, 2)  # Volver a [batch_size, seq_len, input_size]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_size*2]
        
        # Atención o último output
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
            features = context
        else:
            # Usar el último output
            features = lstm_out[:, -1, :]
        
        # Fully connected layers
        features = F.relu(self.fc1(features))
        features = self.dropout1(features)
        logits = self.fc2(features)
        
        return logits

