# import genet
# import torch
from gpt.genet.util.text_character_tokenizer import TextCharacterTokenizer, TextCharacterDataset
from gpt.genet.util.config import TrainConfigure, ModelConfigure
from gpt.genet import Trainer

corpus_file_path = "input.txt"
tokenizer = TextCharacterTokenizer.from_file(corpus_file_path)

config = TrainConfigure(
    vocab_size=tokenizer.vocab_size,
    context_length=128,
    embedding_size=128,
    num_epochs=2,
    batch_size=128,
    num_heads=4,
    num_blocks=4,
)



# model = genet.GPT.from_checkpoint("checkpoints/best_model.ckpt")

dataset = TextCharacterDataset.from_file(corpus_file_path, tokenizer.vocabulary, config.context_length)
trainer = Trainer(config)
trainer.train(dataset)