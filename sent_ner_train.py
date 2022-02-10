from my_loss_functions import *
from nl2tensor import *
from process_control import *
import torch
import torch.nn as nn
import torch.optim
from language_model.transformers import ElectraTokenizer
from nn.embeddings import ElectraModel
from nn.encoder import BiEncoder as BE
from language_model.transformers.configuration_electra import ElectraConfig
from my_optimizers import Ranger
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
import re
import os
from tqdm import trange
import argparse
import json


def set_args(filename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir",
                        default='data/bio_data/',
                        type=str,
                        help="The train data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_data_dir",
                        default='data/bio_data/',
                        type=str,
                        help="The test data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--pretrained_model_config_dir",
                        default='pretrained_model/pytorch_electra_180g_large/large_discriminator_config.json',
                        type=str)
    parser.add_argument("--pretrained_model_dir",
                        default='pretrained_model/pytorch_electra_180g_large/',
                        type=str,
                        help="choose chinese mode.")
    parser.add_argument("--mymodel_save_dir",
                        default='checkpoint/sent_ner/',
                        type=str,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--mymodel_config_dir",
                        default='config/sent_ner_config.json',
                        type=str)
    parser.add_argument("--vocab_dir",
                        default='pretrained_model/pytorch_electra_180g_large/vocab.txt',
                        type=str,
                        help="The vocab data dir.")
    parser.add_argument("--max_sent_len",
                        default=128,
                        type=int,
                        help="句子最大字符数")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--no_gpu",
                        default=False,
                        action='store_true',
                        help="用不用gpu")
    parser.add_argument("--seed",
                        default=6,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--test_size",
                        default=.0,
                        type=float,
                        help="验证集大小")
    parser.add_argument("--weight_decay",
                        default=.0,
                        type=float)
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true')
    parser.add_argument("--train_epochs",
                        default=25,
                        type=int,
                        help="训练次数大小")
    parser.add_argument("--embeddings_lr",
                        default=1e-2,
                        type=float,
                        help="Embeddings初始学习步长")
    parser.add_argument("--encoder_lr",
                        default=5e-4,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float)
    parser.add_argument("--num_layers",
                        default=2,
                        type=int)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--START_TAG",
                        default="[CLS]",
                        type=str,
                        help="序列开始标识符")
    parser.add_argument("--STOP_TAG",
                        default="[SEP]",
                        type=str,
                        help="序列停止标识符")
    parser.add_argument("--MASK_TAG",
                        default="[MASK]",
                        type=str,
                        help="序列掩码符号")
    parser.add_argument("--tag_to_ix",
                        default={"B-Time": 0, "I-Time": 1, "B-Location": 2, "I-Location": 3, "B-Object": 4,
                                 "I-Object": 5, "B-Participant": 6, "I-Participant": 7, "B-Means": 8, "I-Means": 9,
                                 "B-Denoter": 10, "I-Denoter": 11, "o": 12, '[CLS]': 13, "[SEP]": 14, "[MASK]": 15},
                        type=list)
    parser.add_argument("--tag_to_score",
                        default={0: 20, 1: 20, 2: 20, 3: 20, 4: 20,
                                 5: 20, 6: 20, 7: 20, 8: 20,
                                 9: 20, 10: 20, 11: 20,
                                 12: 20, 13: 20, 14: 20, 15: 20},
                        type=list)
    args = parser.parse_args()
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args


try:
    args = set_args('config/sent_ner_args.txt')
except FileNotFoundError:
    args = set_args('config/sent_ner_args.txt')
logger = get_logger()
set_environ()
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    torch.distributed.init_process_group(backend='nccl')


# 定义一个计算准确率的函数
def accuracy(preds, labels, seq_len):
    count, right = 0, 0
    for pred, label in zip(preds, labels):
        for i in range(seq_len):
            if label[i] != len(args.tag_to_ix)-1 and label[i] != len(args.tag_to_ix)-2 \
                    and label[i] != len(args.tag_to_ix)-3:
                count += 1
                _, p = pred[i].topk(1)
                if int(label[i]) == p[0].item():
                    right += 1
    return right/count


# bio转label
def bio2label(t_label, max_len):
    t_label.append(args.STOP_TAG)
    t_label.append(args.STOP_TAG)
    while len(t_label) < max_len:
        t_label.append(args.MASK_TAG)
    return [args.tag_to_ix[t] for t in t_label]


# CRF网络
class NerModel(nn.Module):
    def __init__(self, config):
        super(NerModel, self).__init__()
        self.tagset_size = len(args.tag_to_ix)
        self.encoder = BE(config.hidden_size,
                          args.max_sent_len,
                          config.num_hidden_layers,
                          config.num_attention_heads,
                          1)
        self.dense = nn.Linear(config.hidden_size*2, self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.25)
        self.loss1 = CrossEntropyLoss()
        self.loss2 = SoftenLoss(len(args.tag_to_ix))

    def dynamic_target(self, x, tags):
        ys = torch.ones_like(x)
        s = torch.zeros(self.tagset_size)
        sum = args.max_sent_len
        for tag in tags:
            for i in range(sum):
                tmp = tag[i].item()
                s[tmp] += 1
        for y, tag in zip(ys, tags):
            for i in range(sum):
                tmp = tag[i].item()
                y[i][tmp] = args.tag_to_score[tmp]**(1-s[tmp]/sum)
        return self.soft(ys)

    def test(self, x, y, mask):
        x = self.encoder(x, mask)
        x = self.dense(x)
        x = self.soft(x)
        acc = accuracy(x, y.detach().cpu().numpy(), args.max_text_len)
        return acc

    def get_tag_seq(self, x):
        x = self.encoder(x)
        x = self.dense(x)
        return x

    def forward(self, x, y, mask):
        x = self.drop(x)
        x = self.encoder(x, mask)
        x = self.dense(x)
        x = self.soft(x)
        acc = accuracy(x, y.detach().cpu().numpy(), args.max_sent_len)
        y = self.dynamic_target(x, y)
        return 0.8*self.loss1(x, y) + 0.2*self.loss2(x), acc


def mymodel_train(args, logger, train_dataloader, validation_dataloader):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    embedding = ElectraModel(config=config)
    model = NerModel(config=config)
    try:
        output_model_file = os.path.join(args.mymodel_save_dir, 'embedding/')
        model_state_dict = torch.load(os.path.join(output_model_file, 'pytorch_model.bin'))
        embedding.load_state_dict(model_state_dict)
    except OSError:
        embedding.from_pretrained(os.path.join(args.pretrained_model_dir, 'pytorch_model.bin'), config=config)
        print("PretrainedEmbeddingNotFound")
    try:
        output_model_file = os.path.join(args.mymodel_save_dir, "mymodel.bin")
        model_state_dict = torch.load(output_model_file)
        model.load_state_dict(model_state_dict)
    except OSError:
        print("PretrainedMyModelNotFound")
    if args.fp16:
        embedding.half()
        model.half()
    embedding.to(device)
    model.to(device)
    param_optimizer1 = list(embedding.named_parameters())
    param_optimizer2 = list(model.named_parameters())
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in ['embeddings'])],
         'weight_decay_rate': args.weight_decay,
         'lr': args.embeddings_lr},
        {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in ['embeddings'])],
         'lr': args.encoder_lr},
    ]
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in param_optimizer2 if any(nd in n for nd in ['encoder'])],
         'lr': args.encoder_lr},
        {'params': [p for n, p in param_optimizer2 if not any(nd in n for nd in ['encoder'])],
         'lr': args.learning_rate},
    ]
    optimizer1 = Ranger(optimizer_grouped_parameters1)
    optimizer2 = Ranger(optimizer_grouped_parameters2)
    epochs = args.train_epochs
    bio_records = []
    train_loss_set = []
    acc_records = []
    embedding.train()
    model.train()
    for _ in trange(epochs, desc='Epochs'):
        tr_loss = 0
        eval_loss, eval_accuracy = 0, 0
        nb_tr_steps = 0
        nb_eval_steps = 0
        tmp_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            text_embedding = embedding(input_ids=b_input_ids.squeeze(1).long(),
                                       attention_mask=b_input_mask)
            loss, tmp_eval_accuracy = model(text_embedding, b_labels, b_input_mask.squeeze(1))
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
            tr_loss += loss.item()
            nb_tr_steps += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            tmp_loss.append(loss.item())
        adjust_learning_rate(optimizer1, 0.9)
        adjust_learning_rate(optimizer2, 0.9)
        try:
            train_loss_set.append(tr_loss / nb_tr_steps)
            logger.info('mymodel训练损失:{:.2f},准确率为：{:.2f}%'
                        .format(tr_loss / nb_tr_steps, 100 * eval_accuracy / nb_eval_steps))
            acc_records.append(eval_accuracy / nb_eval_steps)
            bio_records.append(np.mean(train_loss_set))
        except ZeroDivisionError:
            logger.info("错误！请降低batch大小")
        embedding_to_save = embedding.module if hasattr(embedding, 'module') else embedding
        torch.save(embedding_to_save.state_dict(),
                   os.path.join(os.path.join(args.mymodel_save_dir, 'embedding/'), 'pytorch_model.bin'))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.mymodel_save_dir, "mymodel.bin"))
    return embedding, model


def mymodel_test(logger, test_dataloader):
    config = ElectraConfig.from_pretrained(args.mymodel_config_dir)
    embedding = ElectraModel(config=config)
    model = NerModel(config=config)
    output_model_file = os.path.join(args.mymodel_save_dir, 'embedding/')
    model_state_dict = torch.load(os.path.join(output_model_file, 'pytorch_model.bin'))
    embedding.load_state_dict(model_state_dict)
    output_model_file = os.path.join(args.mymodel_save_dir, "mymodel.bin")
    model_state_dict = torch.load(output_model_file)
    model.load_state_dict(model_state_dict)
    if args.fp16:
        embedding.half()
        model.half()
    embedding.to(device)
    model.to(device)
    embedding.eval()
    model.eval()
    acc_records = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            text_embedding = embedding(input_ids=b_input_ids.squeeze(1).long(),
                                       attention_mask=b_input_mask)
            tmp_eval_accuracy = model.test(text_embedding, b_labels, b_input_mask.squeeze(1))
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    try:
        logger.info('准确率为：{:.2f}%'
                    .format(100 * eval_accuracy / nb_eval_steps))
        acc_records.append(eval_accuracy / nb_eval_steps)
    except ZeroDivisionError:
        logger.info("错误！请降低batch大小")
    return acc_records


def get_dataloader(filenames):
    tokenizer = ElectraTokenizer.from_pretrained(args.vocab_dir)
    input_ids = []
    mask_ids = []
    labels = []
    cnt = 0
    Text = re.compile('.+\t')
    Label = re.compile('\t.+')
    text = ""
    t_label = ['[CLS]']
    for line in read_lines(filenames):
        if line != "":
            text += re.sub(u'\t', '', Text.findall(line)[0])
            t_label.append(re.sub(u'\t', '', Label.findall(line)[0]))
        if line == "":
            tmp1, tmp2, _ = text2ids(tokenizer, text, args.max_sent_len)
            label = bio2label(t_label, args.max_sent_len)
            input_ids.append(tmp1)
            mask_ids.append(tmp2)
            labels.append(label)
            cnt += 1
            text = ""
            t_label = ['[CLS]']
    train_input, validation_input, train_mask, validation_mask, train_labels, validation_labels = \
        train_test_split(input_ids, mask_ids, labels, random_state=args.seed, test_size=args.test_size)

    # 将训练集tensor并生成dataloader
    batch_size = args.train_batch_size
    train_inputs = torch.Tensor(train_input)
    train_masks = torch.Tensor(train_mask)
    train_labels = torch.LongTensor(train_labels)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    if args.test_size > 0:
        # 将验证集tensor并生成dataloader
        validation_inputs = torch.Tensor(validation_input)
        validation_masks = torch.Tensor(validation_mask)
        validation_labels = torch.LongTensor(validation_labels)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        return train_dataloader, validation_dataloader
    else:
        return train_dataloader, _


if __name__ == "__main__":
    train_dataloader, validation_dataloader = get_dataloader(args.train_data_dir + "train.txt")
    embedding, model = mymodel_train(args, logger, train_dataloader, validation_dataloader)
    test_dataloader, _ = get_dataloader(args.test_data_dir + "test.txt")
    acc_records = mymodel_test(logger, test_dataloader)