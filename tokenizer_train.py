from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# 定义模型
tokenizer = Tokenizer(models.BPE())  # 可以换成 models.Unigram() 等

# 定义预分词方式
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # 按空格分词
# 中文可以用 pre_tokenizers.Metaspace 或自己写分词器

# 定义特殊 token
special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

# 定义训练器
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=special_tokens)

# 开始训练
files = ["input_tokenizer.txt"]  # 你的文本语料
tokenizer.train(files, trainer)

# 加上 post_processor 来处理 BOS/EOS
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))]
)

# 保存 tokenizer
tokenizer.save("my_tokenizer.json")
