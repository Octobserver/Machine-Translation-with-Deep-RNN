import torch
import torch.nn as nn
from Variables import PADD_index

# the Decoder class
class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, output_size, batch_size, max_length, device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = 1
        self.dropout = nn.Dropout(self.dropout_p)
        self.device = device

        self.embedder = torch.nn.Embedding(output_size, hidden_size, padding_idx = 2, device=device)
        self.Ua = torch.nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False, device=device)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size * 2, device=device)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size * 2)
        self.out = nn.Linear(hidden_size * 5, self.output_size, device=device)

        # additonal lstm layers with residual connections
        self.lstm1 = nn.LSTM(hidden_size * 2, hidden_size * 2)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size * 2)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size * 2)

    # word_batch: [1, batch_size]
    # encoder_outputs: [max_length, batch_size, 2 * hidden_size (bidirectional)]
    # h , c =[]
    def forward(self, word_batch, encoder_outputs, input_batch, h = None, c = None):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        if h is None or c is None:
          h, c = self.get_initial_hidden_state()

        # comput E * w_j
        #embedded = [batch_size, hidden_size]
        embedded = self.embedder(word_batch).squeeze()
        embedded = self.dropout(embedded)

        # Attention 

        # component1: [batch_size, 1, 2 * hidden_size]
        attn_component1 = torch.transpose(h[3], 0, 1)
        # component2: [batch_size, max_length, 2 * hidden_size]
        attn_component2 = self.Ua(torch.transpose(encoder_outputs, 0, 1))
        #m: [batch_size, max_length]
        m = attn_component1.bmm(torch.transpose(attn_component2, 1, 2)).squeeze()

        # aji's
        # attn_scores: [batch_size , max_length]
        attn_scores = self.mask(m, torch.transpose(input_batch, 0, 1))

        # attn_weights: [batch_size, max_length]
        attn_weights = torch.nn.functional.softmax(attn_scores, -1)
        # context: [batch_size, 1, max_length] * [batch_size, max_length, 2 * hidden]
        context = attn_weights.unsqueeze(1).bmm(torch.transpose(encoder_outputs, 0, 1))

        #LSTM updates
        #embedded = [1, batch_size, hidden_size]
        #context = [1, batch_size, 2*hidden_size]
        embedded = embedded.unsqueeze(0)
        context = torch.transpose(context, 0, 1)
        lstm_input = self.attn_combine(torch.cat((embedded, context), -1))
        output, (h1, c1) = self.lstm(lstm_input, (h[0], c[0]))

        #h_t = [1, batch_size, 2*hidden_size]
        output, (h2, c2) = self.lstm1(h1, (h[1], c[1]))

        output, (h3, c3) = self.lstm2(h2, (h[2], c[2]))
        current_input = h2 + h3
        output, (h4, c4) = self.lstm3(current_input, (h[3], c[3]))

        #LSTM outputs
        output = self.out(torch.cat((output, context, embedded), -1)).squeeze()
        output = torch.nn.functional.softmax(output, dim=-1)
        # output: [batch_size * output_size]
        return output, (torch.cat((h1, h2, h3, h4), 0).unsqueeze(1), torch.cat((c1, c2, c3, c4), 0).unsqueeze(1)), attn_weights

    #input_batch: [batch_size, max_length]
    def mask(self, scores, input_batch):
      for i in range(self.batch_size):
        for j in range(self.max_length):
          if input_batch[i][j].item() == PADD_index:
            scores[i][j] = -float('inf')
      return scores

    def get_initial_hidden_state(self):
        return torch.zeros(4, self.num_layers, self.batch_size, self.hidden_size * 2, device=self.device), torch.zeros(4, self.num_layers, self.batch_size, self.hidden_size * 2, device=self.device)
