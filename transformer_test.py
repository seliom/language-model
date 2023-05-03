import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformer import TransformerModel, model_save_path, data_process
from transformer import generate_square_subsequent_mask, get_batch, batchify
from transformer import emsize, d_hid, nlayers, nhead, dropout, bptt, batch_size
from transformer import eval_batch_size


def display_words(tensor, vocab):
    indices = tensor.tolist()  
    num_rows, num_cols = tensor.shape
    
    word_matrix = ""
    for row in range(num_rows):
        for col in range(num_cols):
            word = vocab.lookup_token(indices[row][col])
            word_matrix += "{:<12}".format(word)  
        word_matrix += "\n"
    return word_matrix

def report(model, data_source, vocab, data_index=0):

    model.eval()

    with torch.no_grad():

        src_mask = generate_square_subsequent_mask(bptt)

        inputs, _ = get_batch(data_source, data_index)

        print("Input words:")
        print(inputs)
        print(display_words(inputs, vocab))

        # This is where we could build an actual response to a prompt
        '''
        seq_len = inputs.size(0)

        if seq_len != bptt:  # only on last batch
            src_mask = generate_square_subsequent_mask(seq_len).to(device)

        output = model(inputs, src_mask)
        '''

def main():

    print('Loading vocabulary from dataset')
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    ntokens = len(vocab)
    print(ntokens)

    print('Loading model from saved file ' + model_save_path)
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    model.load_state_dict(torch.load(model_save_path))

    report(model, val_data, vocab)

main()
