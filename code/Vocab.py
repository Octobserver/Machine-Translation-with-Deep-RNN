from Variables import SOS_index, EOS_index, PADD_token, PADD_index, SOS_token, EOS_token
from Util import split_lines

class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = { SOS_index: SOS_token, EOS_index: EOS_token, PADD_index: PADD_token}
        self.n_words = 3  # Count  SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def make_vocabs(src_lang_code, tgt_lang_code, train_file):
        """ Creates the vocabs for each of the langues based on the training corpus.
        """
        src_vocab = Vocab(src_lang_code)
        tgt_vocab = Vocab(tgt_lang_code)

        train_pairs = split_lines(train_file)

        for pair in train_pairs:
            src_vocab.add_sentence(pair[0])
            tgt_vocab.add_sentence(pair[1])

        return src_vocab, tgt_vocab
