from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from tqdm import tqdm
from rouge import Rouge
import json, torch
import numpy as np
import torch.nn.functional as F

def rouge(pred, tar):
    rouge = Rouge()  
    rouge_score = rouge.get_scores(pred, tar, avg=True)
    r1 = rouge_score["rouge-1"]['f']
    r2 = rouge_score["rouge-2"]['f']
    rl = rouge_score["rouge-l"]['f']
    return r1, r2, rl

if __name__ == '__main__':
    enc_char2id_path = 'ckpt/enc_char2id.json'
    dec_char2id_path = 'ckpt/dec_char2id.json'
    # attention_s
    model_path = 'ckpt/attention_s.pt'
    with open('data/seq2seq_dataset.json', 'r', encoding='utf8') as f:
        raw = json.load(f)
    test_dict = defaultdict(list)

    test_dict = sorted(test_dict.items(), key=lambda x: len(x[1]), reverse=True)
    
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    rouge1 = []
    rouge2 = []
    rougel = []
    unigram = []
    bigram = []
    
            
    for i in tqdm(range(1, 101)):
        cur_d = raw[-i]
        txt = cur_d['text']
        scr = [cur_d['diversity_score'], cur_d['central_score'], cur_d['nov_score'], cur_d['fluent_score']]
        cmt = cur_d['cmts'][0]
        cmt_in = cmt[:4]
        target_text = [[x for x in cmt[len(cmt_in):]]]
        prediction = predict(enc_char2id_path, dec_char2id_path, model_path, txt, scr, cmt_in)
        print(prediction)
        prediction = prediction[len(cmt_in):]
        unigram += [c for c in prediction]
        bigram += [prediction[i:i + 2] for i in range(len(prediction) - 1)]
        bleu1.append(sentence_bleu(target_text, [x for x in prediction], (1, 0, 0, 0)))
        bleu2.append(sentence_bleu(target_text, [x for x in prediction], (0, 1, 0, 0)))
        bleu3.append(sentence_bleu(target_text, [x for x in prediction], (0, 0, 1, 0)))
        bleu4.append(sentence_bleu(target_text, [x for x in prediction], (0, 0, 0, 1)))
        r1, r2, rl = 0, 0, 0
        for l in target_text:
            tr1, tr2, trl = rouge([' '.join([x for x in prediction])], [' '.join(l)])
            r1 = max(r1, tr1)
            r2 = max(r2, tr2)
            rl = max(rl, trl)
        rouge1.append(r1)
        rouge2.append(r2)
        rougel.append(rl) 


    def mean(l):
        return sum(l) / len(l)
    
    print('BLEU-1:', mean(bleu1))
    print('BLEU-2:', mean(bleu2))
    print('BLEU-3:', mean(bleu3))
    print('BLEU-4:', mean(bleu4))
    print('ROUGE-1:', mean(rouge1))
    print('ROUGE-2:', mean(rouge2))
    print('ROUGE-L:', mean(rougel))
    print('Dist-1:', len(set(unigram)) / len(unigram))
    print('Dist-2:', len(set(bigram)) / len(bigram))
        