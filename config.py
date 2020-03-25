import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--hidden_size", type=int, default=500)
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--batch_size", type=int, default = 32)
parser.add_argument("--dropout", type=int, default = 0.2)
parser.add_argument("--learning_rate", type=int, default = 3e-4)
parser.add_argument("--mode", type=str, default = "train")
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--pretrain_model", type=str, default = "save/2-2_500/")


arg = parser.parse_args()
print(arg)
hidden_size= arg.hidden_size
dropout = arg.dropout
batch_size = arg.batch_size
learning_rate=arg.learning_rate
save_path = arg.save_path
mode = arg.mode
max_grad_norm = arg.max_grad_norm
pretrain_model = arg.pretrain_model

use_pretrain = True

iterations = 60000
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 40
MIN_COUNT = 5

