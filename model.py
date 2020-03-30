import torch
import torch.nn as nn
import torch.optim as optim
from dataLoader import getBatchData,voc,pairs
import os
import random
import torch.nn.functional as F
import jieba
import re
import config
# import warnings
# warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = config.save_path
hidden_size = config.hidden_size
encoder_n_layers = 2
decoder_n_layers = 2
dropout = config.dropout
batch_size = config.batch_size
MAX_LENGTH = config.MAX_LENGTH
learning_rate = config.learning_rate
decoder_learning_ratio = 5.0
iterations = config.iterations

class EncoderRNN(nn.Module):
	def __init__(self,hidden_size,embedding,n_layers,dropout):
		super(EncoderRNN,self).__init__()
		self.hidden_size = hidden_size
		self.embedding = embedding

		self.gru = nn.GRU(hidden_size,hidden_size,n_layers,
			              dropout=dropout,bidirectional=True)

	def forward(self,input_seq,input_lengths,hidden = None):
		embedded = self.embedding(input_seq)  #所以其实是通过词的index去embedding里面找
		packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,enforce_sorted = False)
		outputs, hidden = self.gru(packed, hidden)
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
		outputs = outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
		return outputs,hidden

class Attention(nn.Module):
	"""docstring for Attention"""
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
		self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self,hidden,encoder_output):
		attention = self.concat_score(hidden,encoder_output)
		attention = attention.t() #转置 为什么要？
		return F.softmax(attention,dim=1).unsqueeze(1)


class AttenDecoderRNN(nn.Module):
	"""docstring for AttenDecoderRNN"""
	def __init__(self,embedding,hidden_size,output_size, n_layers=1, dropout=0.1):
		super(AttenDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout
		
		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers == 1 else dropout))
		self.concat = nn.Linear(hidden_size*2,hidden_size)
		self.out = nn.Linear(hidden_size,output_size)
		self.attn = Attention(hidden_size)

	def forward(self,input_step,last_hidden,encoder_output):
		input_step = input_step.to(device)
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)
		

		#这里获取embedding之后传给gru了

		rnn_output,hidden = self.gru(embedded,last_hidden)

		attn_weights = self.attn(rnn_output,encoder_output)
		context = attn_weights.bmm(encoder_output.transpose(0,1))
		# print("context:",context)

		rnn_output = rnn_output.squeeze(0)  #去除只有一维的
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))

		output = self.out(concat_output)
		output = F.softmax(output, dim=1)
		return output, hidden


def maskNLLLoss(inp, target, mask):
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
	loss = crossEntropy.masked_select(mask).mean()
	loss = loss.to(device)
	return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, 
	decoder, embedding,encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	input_variable = input_variable.to(device)
	lengths = lengths.to(device)
	target_variable = target_variable.to(device)
	mask = mask.to(device)

	loss = 0
	print_losses = []
	n_totals = 0

	encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

	decoder_input = torch.LongTensor([[config.SOS_token for _ in range(batch_size)]])
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	# use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	for t in range(max_target_len):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
		_, topi = decoder_output.topk(1) #使用的是解码器当前的输出
		decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
		decoder_input = decoder_input.to(device)
		mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
		loss += mask_loss
		print_losses.append(mask_loss.item() * nTotal)
		n_totals += nTotal
	loss.backward()
	torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
	torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.max_grad_norm)

	#梯度更新
	encoder_optimizer.step()
	decoder_optimizer.step()

	return sum(print_losses) / n_totals


def trainIters(voc, pairs, embedding, encoder_n_layers, decoder_n_layers):
	training_batches = [getBatchData(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(iterations)]
	#training_batches为全部训练数据  每batch_size大小为一批
	print('Initializing ...')
	print_loss = 0
	print("Training...")

	encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
	decoder = AttenDecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

	if config.use_pretrain:
		checkpoint = torch.load(os.path.join(config.pretrain_model,'model20000.pth'))
		encoder.load_state_dict(checkpoint['en'])
		decoder.load_state_dict(checkpoint['de'])

	# checkpoint = torch.load('save/2-2_500/model20000.pth')
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	encoder.train()
	decoder.train()
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

	for iteration in range(20001, iterations + 1):
		training_batch = training_batches[iteration - 1]
		input_variable, lengths, target_variable, mask, max_target_len = training_batch
		loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
			decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
		print_loss += loss
		#输出每50个iteation的平均loss
		if iteration % 500 == 0:
			print_loss_avg = print_loss / 500
			print("Iteration: {}; Average loss: {:.4f}".format(iteration, print_loss_avg))
			print_loss = 0

		if (iteration % 10000 == 0):
			# directory = os.path.join(save_path, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			state = {'en': encoder.state_dict(),'de': decoder.state_dict(), 'iteration': iteration,'loss': loss}
			dir = os.path.join(save_path, 'model{}.pth'.format(iteration))
			torch.save(state,dir)
			# torch.save({
			# 	'iteration': iteration,
			# 	'en': encoder.state_dict(),
			# 	'de': decoder.state_dict(),
			# 	'en_opt': encoder_optimizer.state_dict(),
			# 	'de_opt': decoder_optimizer.state_dict(),
			# 	'loss': loss,
			# 	'voc_dict': voc.__dict__,
			# 	'embedding': embedding.state_dict()
			# 	},os.path.join(directory, '{}_{}'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
	def  __init__(self,encoder,decoder):
		super(GreedySearchDecoder,self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self,input_seq,input_length,max_length):
		encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
		decoder_hidden = encoder_hidden[:decoder.n_layers]
		decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * config.SOS_token

		all_tokens = torch.zeros([0], device=device, dtype=torch.long)
		all_scores = torch.zeros([0], device=device)

		for _ in range(max_length):
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
			all_scores = torch.cat((all_scores, decoder_scores), dim=0)
			decoder_input = torch.unsqueeze(decoder_input, 0)

		return all_tokens, all_scores



def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
	indexes_batch = [voc.word2index[word] for word in sentence] + [config.EOS_token]
	input_batch = [indexes_batch]
	lengths = torch.tensor([len(indexes) for indexes in input_batch])
	input_batch = torch.LongTensor([indexes_batch]).transpose(0, 1)
	input_batch = input_batch.to(device)
	lengths = lengths.to(device)
	tokens, scores = searcher(input_batch, lengths, max_length)
	decoded_words = [voc.index2word[token.item()] for token in tokens]
	return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
	input_sentence = ''
	while(1):
		try:
			input_sentence = input('>')
			if input_sentence == 'q' or input_sentence == 'quit':
				break
			# input_sentence = normalizeString(input_sentence)
			cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
			input_seq = jieba.lcut(cop.sub("",input_sentence))
			# print(input_seq)
			# print(input_seq)
			# input_seq = [voc.word2index[word] for word in input_seq]
			# print(input_seq)
			output_words = evaluate(encoder, decoder, searcher, voc, input_seq)
			# print(output_words)
			output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
			print('Bot:', ''.join(output_words))
		except KeyError:
			print("发生错误")

		

# print('Building encoder and decoder ...')
embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = AttenDecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
encoder = encoder.to(device)
decoder = decoder.to(device)
# print('Models built and ready to go!')
# print(voc.num_words)



def ModelTrain():
	print("start train")
	trainIters(voc, pairs,embedding, encoder_n_layers, decoder_n_layers)

def ModelVal():
	with torch.no_grad():
		checkpoint = torch.load(os.path.join('save/','model100000.pth'),map_location=torch.device('cpu'))
		encoder.load_state_dict(checkpoint['en'])
		decoder.load_state_dict(checkpoint['de'])
		encoder.eval()
		decoder.eval()
		searcher = GreedySearchDecoder(encoder,decoder)
		evaluateInput(encoder,decoder,searcher,voc)

