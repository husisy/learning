import os
import sentencepiece as spm

hf_data = lambda *x: os.path.join('data', *x)
# wget https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


## example00
tmp0 = hf_data('botchan.txt')
model_path = hf_file('test00.model')
tmp1 = model_path.rsplit('.',1)[0]
spm.SentencePieceTrainer.train(f'--input={tmp0} --model_prefix={tmp1} --vocab_size=2000')
sp = spm.SentencePieceProcessor(model_file=model_path)

sentence = 'This is a test'
sp.encode(sentence, out_type=str) #['▁This', '▁is', '▁a', '▁t', 'est']
sp.encode(sentence, out_type=int) #[212, 32, 10, 587, 446]
sp.decode(sp.encode(sentence, out_type=str))
sp.decode(sp.encode(sentence, out_type=int))

sp.get_piece_size() #2000

sp.piece_to_id('▁This')
sp.id_to_piece(sp.piece_to_id('▁This'))
sp.piece_to_id('__MUST_BE_UNKNOWN__') #0
ord('▁') #9601
ord('_') #95

# <unk>: 0
# <s>: 1, control symbol
# </s>: 2, control symbol
for x in range(3):
    print(sp.id_to_piece(x), sp.is_control(x))


## example01
tmp0 = hf_data('botchan.txt')
model_path0 = hf_file('example01_0.model')
tmp1 = model_path0.rsplit('.',1)[0]
spm.SentencePieceTrainer.train(f'--input={tmp0} --model_prefix={tmp1} --user_defined_symbols=<sep>,<cls> --vocab_size=2000')
sp0 = spm.SentencePieceProcessor(model_file=model_path0)

model_path1 = hf_file('example01_1.model')
tmp1 = model_path1.rsplit('.',1)[0]
spm.SentencePieceTrainer.train(f'--input={tmp0} --model_prefix={tmp1} --vocab_size=2000')
sp1 = spm.SentencePieceProcessor(model_file=model_path1)

sentence = 'this is a test<sep> hello world<cls>'
sp0.encode(sentence, out_type=str) #['▁this', '▁is', '▁a', '▁t', 'est', '<sep>', '▁he', 'll', 'o', '▁world', '<cls>']
sp1.encode(sentence, out_type=str) #['▁this', '▁is', '▁a', '▁t', 'est', '<', 'se', 'p', '>', '▁he', 'll', 'o', '▁world', '<', 'c', 'l', 's', '>']
sp0.piece_to_id('<sep>') #3
sp0.piece_to_id('<cls>') #4
sp0.id_to_piece(3) #'<sep>'
sp0.id_to_piece(4) #'<cls>'
sp0.decode([3]) #'<sep>'
sp0.decode([4]) #'<cls>'

tmp0 = hf_data('botchan.txt')
model_path2 = hf_file('example01_2.model')
tmp1 = model_path2.rsplit('.',1)[0]
spm.SentencePieceTrainer.train(f'--input={tmp0} --model_prefix={tmp1} --control_symbols=<sep>,<cls> --vocab_size=2000')
sp2 = spm.SentencePieceProcessor(model_file=model_path2)

sentence = 'this is a test<sep> hello world<cls>'
sp2.encode(sentence, out_type=str) #['▁this', '▁is', '▁a', '▁t', 'est', '<', 'se', 'p', '>', '▁he', 'll', 'o', '▁world', '<', 'c', 'l', 's', '>']
sp2.piece_to_id('<sep>') #3
sp2.piece_to_id('<cls>') #4
sp2.id_to_piece(3) #'<sep>'
sp2.id_to_piece(4) #'<cls>'
sp2.decode([3]) #''
sp2.decode([4]) #''

print(sp2.piece_to_id('<sep>'))  # 3
print(sp2.piece_to_id('<cls>'))  # 4
print('3=', sp2.decode_ids([3]))  # decoded to empty
print('4=', sp2.decode_ids([4]))  # decoded to empty


#TODO https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
