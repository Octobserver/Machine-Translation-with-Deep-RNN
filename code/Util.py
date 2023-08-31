import sys
import math
import torch
import random
import heapq
import logging
import matplotlib.pyplot as plt
from Beam import Beam

from Variables import SOS_index, EOS_index, EOS_token, PADD_index, device, MAX_LENGTH

def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def pad(seq, max_length = MAX_LENGTH):
	# pad seq with max_length - len(seq) number of eos index
  seq = seq.tolist()
  for i in range (max_length - len(seq)):
    seq.append([PADD_index])
  return seq


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


def batch(batch_size,src_vocab,tgt_vocab,input_pairs, max_length = MAX_LENGTH):

    # shuffle the input dataset
    random.shuffle(input_pairs)

    source_batch = []
    target_batch = []
    for j in range(batch_size):
      # get the current pair
      pair = input_pairs[random.randint(0, len(input_pairs) - 1)]
      input_tensor = tensor_from_sentence(src_vocab, pair[0])
      target_tensor = tensor_from_sentence(tgt_vocab, pair[1])

      source_batch.append(pad(input_tensor, max_length))
      target_batch.append(pad(target_tensor, max_length))
    
     # convert to tensors and reshape to max_length * batch size
    source_batch= torch.transpose(torch.LongTensor(source_batch).squeeze(), -2, -1)
    target_batch = torch.transpose(torch.LongTensor(target_batch).squeeze(), -2, -1)

    return source_batch, target_batch


def train(input_batch, target_batch, encoder, decoder, optimizer, criterion, batch_size, max_length=MAX_LENGTH):

    # input_batch, target_batch: max_length * batch_size
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    optimizer.zero_grad()
    batched_sentence_loss = 0
    
    # forward, backward = encoder.forward_sentence(input_tensor)
    encoder_outputs, (hn, cn) = encoder.forward(input_batch)

    # change the shape of h_n, c_n
    hn = hn.view(1, batch_size, -1).unsqueeze(0).repeat(4, 1, 1, 1)
    cn = cn.view(1, batch_size, -1).unsqueeze(0).repeat(4, 1, 1, 1)

    # SOS
    SOS_tensor = (torch.ones((1, batch_size)) * SOS_index).type(torch.LongTensor)
    output, (ht, ct), attn_weights = decoder.forward(SOS_tensor, encoder_outputs, input_batch, hn, cn)

    batched_word_loss = criterion( output, target_batch[0])
    batched_sentence_loss += batched_word_loss
    for i in range(target_batch.shape[0] - 1):
       output, (ht, ct), attn_weights = decoder.forward(target_batch[i], encoder_outputs, input_batch, ht, ct)   
       batched_word_loss = criterion( output, target_batch[i+1])
       batched_sentence_loss += batched_word_loss
    
    batched_sentence_loss /= encoder.batch_size
    batched_sentence_loss.backward()
    optimizer.step()

    return batched_sentence_loss.item() 


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """
    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        beam_width = 6 # number of hypotheses to keep
        k = 3

        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_batch = torch.LongTensor(pad(input_tensor)).repeat(1, encoder.batch_size)
        encoder_outputs, (hn, cn) = encoder.forward(input_batch)
        
        decoder_input = (torch.ones((1, encoder.batch_size)) * SOS_index).type(torch.LongTensor)
        # change the shape of h_n, c_n
        ht = hn.view(1, encoder.batch_size, -1).unsqueeze(0).repeat(4, 1, 1, 1)
        ct = cn.view(1, encoder.batch_size, -1).unsqueeze(0).repeat(4, 1, 1, 1)

        decoded_words = []
            
        endnodes = []
        node = Beam((ht, ct), None, torch.LongTensor([[SOS_index]], device=device), 0, 1, None)
        nodes = []
        # push init node
        heapq.heappush(nodes, (0.0, node))
        
        for di in range(max_length):
            best_hypotheses = []

            while len(nodes) != 0:
                # top of heap
                top_node = heapq.heappop(nodes)
                score, n = top_node

                if n.wordid.item() == EOS_index and n.prevNode != None:
                    score = -n.score_full_sentence(0.4, 0.4)
                    score = score if not math.isinf(score) else sys.float_info.max
                    endnodes.append((score, n))
                    continue

                # decoder input
                decoder_input = (torch.ones((1, encoder.batch_size)) * n.wordid.squeeze().detach()).type(torch.LongTensor)
                # decoder output
                decoder_output, (h, c), decoder_attention = decoder.forward(decoder_input, encoder_outputs, input_batch, n.h[0], n.h[1])

                topv, topi = decoder_output.data[0].topk(k)

                # PUT HERE REAL BEAM SEARCH OF TOP
                for i in range(k):
                    prob = topv[i]
                    index = topi[i]
                  
                    log_prob = torch.log10(prob).item()
                    #hidden, previous node, word, logp, length, attn
                    attn =  torch.cat((n.attn_weights, decoder_attention[0].unsqueeze(0)), 0) if (n.attn_weights != None) else decoder_attention[0].unsqueeze(0)
                    new_node = Beam((h, c), n, index, n.logp + log_prob, n.leng + 1, attn)
                    #alpha = beta = 0.4
                    score = -new_node.score_full_sentence(0.4, 0.4) if (di == max_length -1) else -new_node.eval(0.4)
                    score = score if not math.isinf(score) else sys.float_info.max
                    heapq.heappush(best_hypotheses, (score, new_node))
            
            best_hypotheses = heapq.nsmallest(beam_width, best_hypotheses, key=None)

            if (di == max_length -1):
                for h in best_hypotheses:
                    endnodes.append(h)
            else:
                nodes = best_hypotheses
       
        if len(endnodes) == 0:
            score, best_translation = nodes[0]
        else:
            heapq.heapify(endnodes)
            endnodes = heapq.nsmallest(beam_width, endnodes, key=None)
            score, best_translation = heapq.heappop(endnodes)

        attn = best_translation.attn_weights
        while best_translation.prevNode != None:
            decoded_words.append(tgt_vocab.index2word[best_translation.wordid.item()])
            best_translation = best_translation.prevNode

        decoded_words.reverse()
        print(decoded_words)
        return decoded_words, attn

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


# Translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def show_attention(input_sentence, output_words, attentions, attention_plot_index):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    input_sentence = input_sentence.split()
    attentions = attentions[: len(output_words), : len(input_sentence)]
    plt.imshow(attentions, cmap='hot', interpolation='nearest')
    plt.yticks(range(len(output_words)), output_words)
    plt.xticks(range(len(input_sentence)), input_sentence)
    plt.savefig('attention' + str(attention_plot_index) + '.png')


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, attention_plot_index):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, attention_plot_index)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())