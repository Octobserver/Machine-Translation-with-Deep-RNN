import torch
import torch.nn as nn

# the Encoder class
class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size, batch_size, max_length, device):
        super(EncoderRNN, self).__init__()
        # the size of the embedding vector
        self.hidden_size = hidden_size
        # the size of the input word vector
        self.input_size = input_size
        # the size of individual batches
        self.batch_size = batch_size
        # the length of padded sentences
        self.max_length = max_length
        # the number of LSTM layers
        self.num_layers = 1
        # Embedder matrix: [input_size, hidden_size]
        self.embedder = nn.Embedding(self.input_size,self.hidden_size, padding_idx = 2, device=device)
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers=self.num_layers,bidirectional=True)
        # additional LSTM layers with residual connections
        self.lstm1 = nn.LSTM(2 * hidden_size, 2 * hidden_size,num_layers=self.num_layers,bidirectional=False)
        self.lstm2 = nn.LSTM(2 * hidden_size, 2 * hidden_size,num_layers=self.num_layers,bidirectional=False)

        self.device = device
  
    def forward(self, src_batch, h = None, c = None):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        # src_batch: max_length * batch_size
        if (h is None or c is None):
          h, c = self.get_initial_hidden_state()

        #print(h.shape)
        # comput E * w_j
        # embedded: [max_length, batch_size, hidden_size]
        embedded = self.embedder(src_batch)

        # h_n: [2 (bidirectional), batch_size, hidden_size]
        # output: [max_length, batch_size, 2 * hidden_size (bidirectional)]
        output, (h_n, c_n) = self.lstm.forward(embedded, (h, c))

        # residual connections
        # previous_output: [max_length, batch_size, 2 * hidden_size]
        previous_output = output
        h_n = h_n.view(1, self.batch_size, -1)
        c_n = c_n.view(1, self.batch_size, -1)
        # current_output: [max_length, batch_size, 2 * hidden_size]
        current_output, (h_n, c_n) = self.lstm1(previous_output, (h_n, c_n))

        previous_output = previous_output + current_output 
        current_output, (h_n, c_n) = self.lstm2(previous_output, (h_n, c_n))

        return current_output, (h_n, c_n)

    def get_initial_hidden_state(self):
        return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size, device=self.device), torch.zeros(self.num_layers * 2, self.batch_size,  self.hidden_size, device=self.device)
