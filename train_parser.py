import json
import argparse
import glob
import logging
import os
import sys
import random
import timeit
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer, 
                        BertModel, get_linear_schedule_with_warmup, 
                        squad_convert_examples_to_features)
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import pdb
import copy
from tqdm import tqdm, trange
from torch.autograd import Variable
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from datetime import datetime
import os
from utils import readGZip, filter_firstKsents
from torch.utils.data import DataLoader, Dataset
import math
from dateparser import parse

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, )
    ),
    (),
)

MODEL_CLASSES = {"bert": (BertConfig, BertModel, BertTokenizer)}

def isNaN(num):
    return num != num

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class ParserDataset(Dataset):
    def __init__(self, path, data, tokenizer, max_seq_length, option, retain_label=True, shuffle=True):
        super(ParserDataset, self).__init__()
        self.shuffle = shuffle
        self.retain_label = retain_label
        self.max_seq_length = max_seq_length
        self.option = option
        self.data = data
        self.tokenizer = tokenizer
        self.path = path

    def __len__(self):
        return len(self.data)


    def parse(self, structure, question):
        tmp = '[CLS] {} [SEP]'.format(question)
        part1 = self.tokenizer.tokenize(tmp)
        input_type = [0] * len(part1)

        tmp = '{} [SEP]'.format(structure['aggregate'])
        part2 = self.tokenizer.tokenize(tmp)
        input_type += [1] * len(part2)

        tmp = '{} [SEP]'.format(structure['target'])
        part3 = self.tokenizer.tokenize(tmp)
        input_type += [0] * len(part3)

        tmp = '{} is {} [SEP]'.format(structure['where'][0], structure['where'][1])
        part4 = self.tokenizer.tokenize(tmp)
        input_type += [1] * len(part4)

        return part1 + part2 + part3 + part4, input_type


    def __getitem__(self, index):
        if self.shuffle:
            d = random.choice(self.data)
        else:
            d = self.data[index]

        question = d['question']
        input_tokens = []
        input_types = []
        input_masks = []
        labels = []

        if self.retain_label:
            grountruth, _ = self.parse(d['groundtruth'], question)

        info = []
        for a in d['aggregate']:
            for t in d['target']:
                for w in d['where'][:12]:
                    input_token, input_type = self.parse({'aggregate': a, 'target': t, 'where': w}, question)
                    input_tokens.append(input_token)
                    input_types.append(input_type)
                    input_masks.append([1] * len(input_token))
                    info.append((a, t, w))
                    
                    if self.retain_label:
                        if input_token == grountruth:
                            labels.append(1)
                        else:
                            labels.append(0)

        max_len = max([len(_) for _ in input_tokens])
        for i in range(len(input_tokens)):
            input_tokens[i] = input_tokens[i] + ['[PAD]'] * (max_len - len(input_tokens[i]))
            input_tokens[i] = self.tokenizer.convert_tokens_to_ids(input_tokens[i][:self.max_seq_length])
            input_types[i] = (input_types[i] + [0] * (max_len - len(input_types[i])))[:self.max_seq_length ]
            input_masks[i] = (input_masks[i] + [0] * (max_len - len(input_masks[i])))[:self.max_seq_length ]

        input_tokens = torch.LongTensor(input_tokens)
        input_types = torch.LongTensor(input_types)
        input_masks = torch.LongTensor(input_masks)
        labels = torch.FloatTensor(labels)
        
        #labels = torch.LongTensor(d['labels'])
        if self.retain_label:
            return input_tokens, input_types, input_masks, labels
        else:
            return input_tokens, input_types, input_masks, info


class PretrainedModel(nn.Module):
    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.base.config.architectures = [model_to_save.base.__class__.__name__]

        # Save configuration file
        model_to_save.base.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))


class ParserModel(PretrainedModel):
    def __init__(self, model_class, model_name_or_path, config, cache_dir, dim=768):
        super(ParserModel, self).__init__()
        self.base = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir if cache_dir else None,
        )
        self.projection = nn.Sequential(nn.Linear(dim, dim), 
                                        nn.ReLU(),
                                        nn.Linear(dim, 1))

    def forward(self, input_tokens, input_types, input_masks):
        inputs = {"input_ids": input_tokens, "token_type_ids": input_types, "attention_mask": input_masks}
        _, text_representation = self.base(**inputs)
        score = self.projection(text_representation).squeeze()
        probs = torch.softmax(score, 0)
        return probs

def convert2num(string):
    string = string.replace(',', '')
    if string.endswith('%'):
        string = string.rstrip('%')
    try:
        string = float(string)
        return string
    except Exception:
        return None

def find_superlative(table, headers, header, sup_type):
    result = ''
    mapping = []
    for j in range(len(table['header'])):
        if header != headers[j]:
            continue

        if headers[j] not in ['#', 'Type', 'Name', 'Location', 'Position', 'Category', 'Nationality',
                              'School', 'Notes', 'Notability', 'Country']:
            activate_date_or_num = None
            for i, row in enumerate(table['data']):
                if len(table['data'][i][j][0]) > 1:
                    continue

                data = table['data'][i][j][0][0]
                if data in ['', '-']:
                    continue

                num = convert2num(data)
                if num and data.isdigit() and num > 1000 and num < 2020 and activate_date_or_num in ['date', None]:
                    date_format = parse(data)
                    mapping.append((date_format, 'date', [data, (i, j), None, None, 1.0]))
                    activate_date_or_num = 'date'
                elif num and activate_date_or_num in ['num', None]:
                    mapping.append((num, 'number', [data, (i, j), None, None, 1.0]))
                    activate_date_or_num = 'num'
                else:
                    try:
                        date_format = parse(data)
                        if date_format and activate_date_or_num in ['date', None]:
                            mapping.append((date_format, 'date', [data, (i, j), None, None, 1.0]))
                            activate_date_or_num = 'date'
                    except Exception:
                        continue
        
    if len(mapping) <= 2:
        return result
    else:
        tmp = sorted(mapping, key = lambda x: x[0])                
        if sup_type in ['earliest', 'minimum']:
            result = tmp[0][-1][0]
        else:
            result = tmp[-1][-1][0]
        return result

def func(inputs):
    step, d = inputs
    table_id = d['table_id']
    with open('{}/tables_tok/{}.json'.format('../../', table_id), 'r') as f:
        table = json.load(f)

    headers = [_[0][0] for _ in table['header']]

    pred = d['prediction']

    r = ''
    if pred[0] != '':
        r = find_superlative(table, headers, pred[1], pred[0])
    else:
        if pred[2][0] != '*':
            col_idx = headers.index(pred[1])
            con_idx = headers.index(pred[2][0])
            for row in table['data']:
                if ' , '.join(row[con_idx][0]) == pred[2][1]:
                    r = ' , '.join(row[col_idx][0])

    return step, r

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--option",
        default='parser',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )    
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")    
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--resource_dir",
        type=str,
        default='../../',
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )   
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")        
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )   
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="/tmp/",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_execute", action="store_true", help="Whether to run eval on the dev set.")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    
    if args.do_train:
        args.output_dir = args.option
        args.output_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    else:
        assert args.output_dir != None, "You must set an output dir"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = ParserModel(model_class, args.model_name_or_path, config, args.cache_dir)
    model.to(args.device)
    
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_data = readGZip(args.train_file)
        dataset = ParserDataset(args.resource_dir, train_data, tokenizer, args.max_seq_length, args.option, retain_label=True, shuffle=True)
        loader = DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=8, shuffle=False, pin_memory=True)

        tb_writer = SummaryWriter(log_dir=args.output_dir)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        t_total = len(train_data) // args.gradient_accumulation_steps * args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0

        tr_loss, logging_loss = 0.0, 0.0
        model.train()
        model.zero_grad()

        train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                *data, labels = tuple(Variable(t).to(args.device) for t in batch)
                probs = model(*data)
                
                loss = torch.sum(-torch.log(probs + 1e-8) * labels)

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    #if args.local_rank == -1 and args.evaluate_during_training:
                    #    results = evaluate(args, model, tokenizer)
                    #    for key, value in results.items():
                    #        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("{}_lr".format(args.option), scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("{}_loss".format(args.option), (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-epoch{}".format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
        
        tb_writer.close()

    if args.do_eval:
        dev_data = readGZip(args.predict_file)
        model.eval()

        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin')))
        logger.info("Loading model from {}".format(args.output_dir))

        dataset = ParserDataset(args.resource_dir, dev_data, tokenizer, args.max_seq_length, args.option, retain_label=False, shuffle=False)
        loader = DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=8, shuffle=False, pin_memory=True)
        
        output_data = copy.copy(dev_data)
        for step, batch in enumerate(tqdm(loader, desc="Evaluation")):
            data = tuple(Variable(t).to(args.device) for t in batch[:-1])
            probs = model(*data)
            prediction = batch[-1][torch.argmax(probs, 0).item()]

            output_data[step]['prediction'] = prediction

        with open('parser_predictions.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        """
        for model_path in os.listdir(args.output_dir):
            if model_path.startswith('checkpoint'):
                model.load_state_dict(torch.load(os.path.join(args.output_dir, model_path, 'pytorch_model.bin')))
                logger.info("Loading model from {}".format(model_path))
                
                eval_loss = 0
                for step, batch in enumerate(tqdm(loader, desc="Evaluation")):
                    *data, labels = tuple(Variable(t).to(args.device) for t in batch)
                    probs = model(*data)
                    loss = torch.sum(-torch.log(probs + 1e-8) * labels)
                    eval_loss += loss.item()
                eval_loss = eval_loss / len(loader)

                logger.info("{} acheives average loss = {}".format(model_path, eval_loss))
        """
    if args.do_execute:
        with open(args.predict_file, 'r') as f:
            data = json.load(f)

        from multiprocessing import Pool
        pool = Pool(64)

        results = pool.map(func, enumerate(data))

        pool.close()
        pool.join()

        for step, r in results:
            del data[step]['prediction']
            data[step]['pred'] = r

        with open('sql_predictions.json', 'w') as f:
            json.dump(data, f, indent=2)
 

if __name__ == "__main__":
    main()