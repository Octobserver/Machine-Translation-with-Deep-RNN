import torch

# Beam Search Decoder
class Beam(object):
    """the class for the beam search decoder
    """
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, attn_weights):

        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.attn_weights = attn_weights

    def eval(self, alpha = 1.0):
        lpy = (((5 + self.leng) ** alpha) / ((5 + 1) ** alpha))
        return self.logp / lpy
    
    def score_full_sentence(self, alpha = 0.0, beta = 0.0):
        score = self.eval(alpha)
        cp = beta * torch.sum(torch.log10(torch.clamp(torch.sum(self.attn_weights, 0, keepdim=True), max = 1.0, min = 10**-1)), -1).item()
        return score + cp

    def __lt__(self, beam2):
        return self.logp > beam2.logp

    def __le__(self, beam2):
        return self.logp >= beam2.logp