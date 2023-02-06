
import json
import torch
import torch.nn.functional as F
import numpy as np
import random

class Generator:
    def __init__(self, enc_char2id_path, dec_char2id_path, model_path, wordlib_path, device, need_scr, model_class, bidirectional, rnn_type=None):
        with open(enc_char2id_path, encoding='utf8') as f:
            self.enc_char2id = json.load(f)   
        with open(dec_char2id_path, encoding='utf8') as f:
            self.dec_char2id = json.load(f) 
        self.dec_id2char = {}
        with open(wordlib_path, encoding='utf8') as f:
            self.word_lib = json.load(f) 
        for c in self.dec_char2id:
            i = self.dec_char2id[c]
            self.dec_id2char[i] = c
        if rnn_type == None:    
            self.model = model_class(V0=len(self.enc_char2id), V1=len(self.dec_char2id), device=device, need_scr=need_scr)
        else:
            self.model = model_class(V0=len(self.enc_char2id), V1=len(self.dec_char2id), device=device, need_scr=need_scr, rnn_type=rnn_type, bidirectional=bidirectional)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device

    # random sampling
    def evaluate(self, txt, scr, cmt_in, prediction_len=32, temp=0.3, txt_max_len=128):
        model = self.model
        enc_char2id = self.enc_char2id
        dec_char2id = self.dec_char2id
        dec_id2char = self.dec_id2char
        device = self.device
        model.eval()
        cmt_in = [c if c in dec_char2id else '<unk>' for c in cmt_in]        
        predicted_text = cmt_in        
        txt = [c if c in enc_char2id else '<unk>' for c in txt]
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
        return ''.join(predicted_text)
    
    def predict(self, txt, scr, tag):        
        need_neg_words = False
        neg_words = ['不够', '不甚', '不太', '不']
        random.shuffle(neg_words)        
        cur_neg_word = neg_words[0]
        if tag not in self.word_lib:
            tag = tag[:-3] + 'pos'
            need_neg_words = True
        tag_lib = self.word_lib[tag]
        all_s = [key for key in tag_lib]
        random.shuffle(all_s)
        cur_s = all_s[0]
        all_o = tag_lib[cur_s]
        random.shuffle(all_o)
        cur_o = all_o[0]
        if need_neg_words:
            cmt_in = cur_s + cur_neg_word + cur_o
        else:
            cmt_in = cur_s + cur_o
        return self.evaluate(txt, scr, cmt_in)
        
    def generate(self, txt, scr):
        cmt_aspects = [['丰富', '内容充实'], ['结构严谨', '中心突出'], ['文采', '新颖', '深刻', '感情真挚'], ['行文规范', '语言流畅']]
        tags = ['R', 'C', 'L', 'F']
        res = {'采分点': [], '扣分点': []}
        tmp = [['采分点：'], ['扣分点：']]
        start_words = ['本文', '文章', '该文', '此文', '这篇作文', '这篇文章']
        random.shuffle(start_words)
        cur_start_word = start_words[0]
        for i, s in enumerate(scr):
            cur_aspects = cmt_aspects[i]
            random.shuffle(cur_aspects)
            for j in range(len(cur_aspects) // 2):
                if s > 80:                
                    aspect = cur_aspects[j] + '-pos'
                    cmt = self.predict(txt, scr, aspect)
                    tmp[0].append('    ' + aspect[:2] + '上，' + cur_start_word + cmt)
                    res['采分点'].append('(' + tags[i] + ')' + cmt)
                else:
                    aspect = cur_aspects[j] + '-neg'
                    cmt = self.predict(txt, scr, aspect)
                    tmp[1].append('    ' + aspect[:2] + '上，' + cur_start_word + cmt)
                    res['扣分点'].append('(' + tags[i] + ')' + cmt)
        cmts = []
        for x in tmp:
            if len(x) > 1:
                cmts += x        
        return cmts, res



