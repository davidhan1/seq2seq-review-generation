from models.attention import Seq2SeqAttn
from models.rnn_only import Seq2SeqRNN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import json
import random
from tqdm import tqdm



def get_vocab(raw):
    enc_seen = set()
    dec_seen = set()

    for d in raw:
        for c in d['text']:
            if c not in enc_seen:
                enc_seen.add(c)
        for cmt in d['cmts']:
            for c in cmt:
                if c not in dec_seen:
                    dec_seen.add(c)

    enc_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(list(enc_seen))
    dec_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(list(dec_seen))

    print('\nenc vocab len:', len(enc_vocab))
    print(enc_vocab[:100])

    print('\ndec vocab len:', len(dec_vocab))
    print(dec_vocab[:100])

    enc_char2id = {}
    enc_id2char = {}

    for i, x in enumerate(enc_vocab):
        enc_char2id[x] = i
        enc_id2char[i] = x

    dec_char2id = {}
    dec_id2char = {}

    for i, x in enumerate(dec_vocab):
        dec_char2id[x] = i
        dec_id2char[i] = x

    return enc_char2id, enc_id2char, dec_char2id, dec_id2char

def get_loader(batch_size, txt_max_len, cmt_max_len, raw, enc_char2id, dec_char2id, device):
    single_data = []

    for d in raw:
        txt = [enc_char2id[c] for c in d['text']][:txt_max_len - 2]
        txt = [enc_char2id['<sos>']] + txt + [enc_char2id['<eos>']]
        if len(txt) < txt_max_len:
            txt += [enc_char2id['<pad>']] * (txt_max_len - len(txt))

        scr = [d['diversity_score'], d['central_score'], d['nov_score'], d['fluent_score']]

        for cmt in d['cmts']:
            cmt = [dec_char2id[c] for c in cmt][:cmt_max_len - 2]
            cmt = [dec_char2id['<sos>']] + cmt + [dec_char2id['<eos>']] 
            if len(cmt) < cmt_max_len:
                cmt += [dec_char2id['<pad>']] * (cmt_max_len - len(cmt))
            
            single_data.append([txt, scr, cmt])

    random.shuffle(single_data)

    print('\nsingle data:\n', single_data[0])

    loader = []
    batch_txt = []
    batch_scr = []
    batch_cmt_in = []
    batch_cmt_out = []

    for i, (txt, scr, cmt) in enumerate(single_data):
        if i != 0 and i % batch_size == 0:
            loader.append([
                torch.LongTensor(batch_txt).permute(1, 0).to(device),
                torch.LongTensor(batch_scr).permute(1, 0).to(device),
                torch.LongTensor(batch_cmt_in).permute(1, 0).to(device),
                torch.LongTensor(batch_cmt_out).to(device)
            ])

            batch_txt = []
            batch_scr = []
            batch_cmt_in = []
            batch_cmt_out = []
        
        batch_txt.append(txt)
        batch_scr.append(scr)
        batch_cmt_in.append(cmt[:-1])
        batch_cmt_out.append(cmt[1:])

    print('\ndata shapes:', loader[0][0].shape, loader[0][1].shape, loader[0][2].shape, loader[0][3].shape)
    
    return loader

def evaluate(model, txt_max_len, enc_char2id, dec_char2id, dec_id2char, device, txt, scr, cmt_in, prediction_len=32, temp=0.3):
    model.eval()
    
    predicted_text = cmt_in
    
    txt = [enc_char2id[c] for c in txt][:txt_max_len - 2]
    txt = [enc_char2id['<sos>']] + txt + [enc_char2id['<eos>']]
    if len(txt) < txt_max_len:
        txt += [enc_char2id['<pad>']] * (txt_max_len - len(txt))
    
    cur_char_idx = dec_char2id[cmt_in[-1]]

    cmt_in = [dec_char2id['<sos>']] + [dec_char2id[c] for c in cmt_in[:-1]]

    txt = torch.LongTensor(txt).unsqueeze(1).to(device)
    scr = torch.LongTensor(scr).unsqueeze(1).to(device)
    cmt_in = torch.LongTensor(cmt_in).unsqueeze(1).to(device)
    
    _, hidden, enc_out = model(txt, scr, cmt_in)
        
    inp = torch.LongTensor([cur_char_idx]).unsqueeze(1).to(device)

    decoder = model.decoder

    for i in range(prediction_len):
        out, hidden = decoder(inp, hidden, enc_out)
        output_logits = out.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()        
        top_index = np.random.choice(len(dec_char2id), p=p_next)
        inp = torch.LongTensor([top_index]).unsqueeze(1).to(device)
        predicted_char = dec_id2char[top_index]
        if predicted_char == '<eos>':
            break
        predicted_text += predicted_char
    
    return predicted_text

if __name__ == '__main__':
    need_scr = False
    bidirectional = False
    # model_class = Seq2SeqAttn
    model_class = Seq2SeqRNN
    rnn_type = 'LSTM'
    if model_class == Seq2SeqAttn:
        if need_scr:
            name = 'attention_s'
        else:
            name = 'attention'
    if model_class == Seq2SeqRNN:
        if rnn_type == 'LSTM' and bidirectional:
            name = 'BiLSTM'
        else:
            name = rnn_type

    BATCH_SIZE = 32
    TXT_MAX_LEN = 128
    CMT_MAX_LEN = 32
    N_EPOCHS = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    with open('data/seq2seq_dataset.json', 'r', encoding='utf8') as f:
        raw = json.load(f)

    print(raw[0])

    enc_char2id, enc_id2char, dec_char2id, dec_id2char = get_vocab(raw)

    loader = get_loader(BATCH_SIZE, TXT_MAX_LEN, CMT_MAX_LEN, raw, enc_char2id, dec_char2id, device)
    # model = model_class(len(enc_char2id), len(dec_char2id), device, need_scr).to(device)
    model = model_class(len(enc_char2id), len(dec_char2id), device, need_scr, rnn_type=rnn_type, bidirectional=bidirectional).to(device)
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)            
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
    criterion = nn.CrossEntropyLoss(ignore_index=dec_char2id['<pad>'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=5, 
        verbose=True, 
        factor=0.5
    )

    # train
    best = float('inf') 
    losses = []
    for epoch in range(N_EPOCHS):
        pbar = tqdm(loader)
        print('\nepoch', epoch)
        i = 0
        for txt, scr, cmt_in, cmt_out in pbar:
            model.train()
            out, hidden, _ = model(txt, scr, cmt_in)
            loss = criterion(out.permute(0, 2, 1), cmt_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if i % 25 == 0:
                shown_loss = sum(losses) / len(losses)
                scheduler.step(shown_loss)
                losses = []
                if shown_loss < best:
                    best = shown_loss
                    torch.save(model.state_dict(), f'ckpt/{name}.pt')
                txt = '书房是我最喜欢呆的地方，那里有一个一整面墙那么大的书柜，摆满了各种各样的书，其中有科学知识书、童话故事书、小说、拓展阅读书等等，它们是我快乐的精神家园。每当在书海中遨游的时候，我的思想就穿越时间和空间，飞到了神秘的太空、美丽的海底世界、远古的生物群落、童话中的宫殿、神奇的人体内部在小小的书房里，有小鸟的欢唱、有日月的光辉，我常常忘了身在何处，感觉不到时间的流逝。'
                scr = [95, 100, 90, 90]
                cmt_in = '文字平实'
                prediction1 = evaluate(model, TXT_MAX_LEN, enc_char2id, dec_char2id, dec_id2char, device, txt, scr, cmt_in)
                print(prediction1)
            pbar.set_postfix({'loss': shown_loss})  
            i += 1
        
