import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):    
    def __init__(self, V0, Hin0=256, Hout0=256, Hout1=256, dropout=0.2, need_scr=True, rnn_type='LSTM', bidirectional=False):
        super().__init__()
        self.txt_emb = nn.Embedding(V0, Hin0)
        self.rnn_type = rnn_type
        if rnn_type == 'RNN':
            self.enc_rnn = nn.RNN(Hin0, Hout0, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.enc_rnn = nn.GRU(Hin0, Hout0, bidirectional=bidirectional)
        else:
            self.enc_rnn = nn.LSTM(Hin0, Hout0, bidirectional=bidirectional)
        self.need_scr = need_scr
        self.enc_fc = nn.Linear(Hout0+4, Hout1)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, txt, scr):
        # txt = (L0, N)
        # scr = (4, N)
        txt = self.dropout(self.txt_emb(txt)) # (L0, N, Hin0)
        enc_out, enc_hid = self.enc_rnn(txt) # (L0, N, 2*Hout0), (2, N, Hout0)
        scr = scr.permute(1, 0) # (N, 4)
        if self.rnn_type == 'LSTM':
            enc_hid, enc_cell = enc_hid
            dec_hid = torch.tanh(self.enc_fc(torch.cat((enc_hid[-1], scr), dim=1))) # (N, Hout1)
            dec_cell = torch.tanh(self.enc_fc(torch.cat((enc_cell[-1], scr), dim=1))) # (N, Hout1)
            dec_hid = (dec_hid, dec_cell)
        else:
            dec_hid = torch.tanh(self.enc_fc(torch.cat((enc_hid[-1], scr), dim=1))) # (N, Hout1)
        return enc_out, dec_hid # (L0, N, 2*Hout0), (N, Hout1)

class Decoder(nn.Module):
    def __init__(self, V1, rnn_type='LSTM', Hout0=256, Hin1=256, Hout1=256, dropout=0.2):
        super().__init__()
        self.dec_vocab_size = V1
        self.rnn_type = rnn_type
        self.dec_embed = nn.Embedding(V1, Hin1)
        if rnn_type == 'RNN':
            self.dec_rnn = nn.RNN(Hin1, Hout1)
        elif rnn_type == 'GRU':
            self.dec_rnn = nn.GRU(Hin1, Hout1)
        else:
            self.dec_rnn = nn.LSTM(Hin1, Hout1)
        self.dec_fc = nn.Linear(Hout1, V1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_in, dec_hid, enc_out):             
        # dec_in = (1, N)
        # dec_hid = (N, Hout1)
        # enc_out = (L0, N, 2*Hout0)
        dec_in = self.dropout(self.dec_embed(dec_in)) # (1, N, Hin1)
        if self.rnn_type == 'LSTM':
            dec_out, dec_hid = self.dec_rnn(dec_in, (dec_hid[0].unsqueeze(0), dec_hid[1].unsqueeze(0))) # (1, N, Hout1), (1, N, Hout1)
        else:
            dec_out, dec_hid = self.dec_rnn(dec_in, dec_hid.unsqueeze(0)) # (1, N, Hout1), (1, N, Hout1)
        dec_in = dec_in.squeeze(0) # (N, Hin1)
        dec_out = dec_out.squeeze(0) # (N, Hout1)
        if self.rnn_type != 'LSTM':
            dec_hid = dec_hid.squeeze(0) # (N, Hout1)
        else:
            dec_hid = (dec_hid[0].squeeze(0), dec_hid[1].squeeze(0)) # (N, Hout1)
        prediction = self.dec_fc(dec_out).unsqueeze(0) # (1, N, V1)
        return prediction, dec_hid # (1, N, V1), (N, Hout1)

class Seq2SeqRNN(nn.Module):
    def __init__(self, V0, V1, device, need_scr, rnn_type='LSTM', bidirectional=False):
        super().__init__()
        self.device = device
        self.V1 = V1
        self.encoder = Encoder(V0, need_scr=need_scr, rnn_type=rnn_type, bidirectional=bidirectional).to(device)
        self.decoder = Decoder(V1, rnn_type=rnn_type).to(device)
    
    def forward(self, txt, scr, cmt_in):
        # txt = (L0, N)
        # scr = (4, N)
        # cmt_in = (L1, N)
        L1, N = cmt_in.shape
        enc_out, dec_hid = self.encoder(txt, scr) # (L0, N, 2*Hout0), (N, Hout1)
        seq_out = torch.zeros(L1, N, self.V1).to(self.device)
        for i in range(L1):
            dec_in = cmt_in[i].unsqueeze(0)
            dec_out, dec_hid = self.decoder(dec_in, dec_hid, enc_out)
            seq_out[i] = dec_out        
        return (seq_out.permute(1, 0, 2), dec_hid, enc_out) # (N, L1, V1), (N, Hout1), (L0, N, 2*Hout0)