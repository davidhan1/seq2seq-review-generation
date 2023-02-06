from generator import *
from models.attention import Seq2SeqAttn
from models.rnn_only import Seq2SeqRNN

def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    enc_char2id_path = 'ckpt/enc_char2id.json'
    dec_char2id_path = 'ckpt/dec_char2id.json'
    model_path = 'ckpt/GRU.pt'
    wordlib_path = 'data/WordLib.json'
    need_scr = False
    model_class = Seq2SeqAttn
    model_class = Seq2SeqRNN
    rnn_type = 'GRU'
    bidirectional = False
    g = Generator(enc_char2id_path, dec_char2id_path, model_path, wordlib_path, device, need_scr, model_class, bidirectional, rnn_type)
    txt = '书房是我最喜欢呆的地方，那里有一个一整面墙那么大的书柜，摆满了各种各样的书，其中有科学知识书、童话故事书、小说、拓展阅读书等等，它们是我快乐的精神家园。每当在书海中遨游的时候，我的思想就穿越时间和空间，飞到了神秘的太空、美丽的海底世界、远古的生物群落、童话中的宫殿、神奇的人体内部在小小的书房里，有小鸟的欢唱、有日月的光辉，我常常忘了身在何处，感觉不到时间的流逝。'
    scr = [95, 100, 60, 60]
    print(g.generate(txt, scr)[1])

test()