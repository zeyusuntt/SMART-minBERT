import argparse
import random
from itertools import zip_longest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from adam_bregman_optimizer import AdamWBreg
from bert import BertModel
from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data
from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask
from optimizer import AdamW

TQDM_DISABLE = True


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert parameters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # raise NotImplementedError
        self.num_labels = len(config.num_labels)
        self.sentiment = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.paraphrase = nn.Linear(config.hidden_size, 1)
        self.similarity = nn.Linear(config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # params for adversarial training
        self.noise_var = 1e-5
        self.iter_var = 3
        self.lr = 1e-3
        self.lbda = 3
        self.mu = 0.2
        self.beta = 0.9


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # raise NotImplementedError
        pooled_output = self.bert(input_ids, attention_mask)['pooler_output']
        pooled_output = self.dropout(pooled_output)

        return pooled_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # raise NotImplementedError
        output = self.forward(input_ids, attention_mask)
        logits = self.sentiment(output)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # raise NotImplementedError
        input_ids = torch.cat([input_ids_1, input_ids_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)
        output = self.forward(input_ids, attention_mask)
        logits = self.paraphrase(output)
        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        # raise NotImplementedError
        input_ids = torch.cat([input_ids_1, input_ids_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)
        output = self.forward(input_ids, attention_mask)
        logits = self.similarity(output)
        return logits

    # adversarial training
    def adv_loss_sentiment(self, input_ids, attention_mask):
        embed = self.forward(input_ids, attention_mask)
        logits = self.sentiment(embed)
        logits = F.softmax(logits, dim=1)
        noise = generate_noise(embed, epsilon=self.noise_var)
        for step in range(self.iter_var):
            adv_logits = self.sentiment(embed + noise)
            adv_logits = F.softmax(adv_logits, dim=1)
            adv_loss = F.cross_entropy(adv_logits, logits, reduction='sum')
            (delta_grad, ) = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            # eff_delta_grad = delta_grad * self.lr
            delta_grad = noise + delta_grad * self.lr
            noise = self._norm_grad(delta_grad)
            noise = noise.detach()
            noise.requires_grad_()
        adv_logits = self.sentiment(embed + noise)
        adv_logits = F.softmax(adv_logits, dim=1)
        adv_loss = F.cross_entropy(logits, adv_logits, reduction='sum')
        return adv_loss

    def adv_loss_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        input_ids = torch.cat([input_ids_1, input_ids_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)
        embed = self.forward(input_ids, attention_mask)
        logits = self.paraphrase(embed)
        logits = F.softmax(logits, dim=1)

        noise = generate_noise(embed, epsilon=self.noise_var)
        loss_func = nn.BCEWithLogitsLoss(reduction='sum')

        for step in range(self.iter_var):
            adv_logits = self.paraphrase(embed + noise)
            adv_logits = F.softmax(adv_logits, dim=1)
            adv_loss = loss_func(adv_logits, logits.float())
            (delta_grad, ) = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            delta_grad = noise + delta_grad * self.lr
            noise = self._norm_grad(delta_grad)
            noise = noise.detach()
            noise.requires_grad_()
        adv_logits = self.paraphrase(embed + noise)
        adv_logits = F.softmax(adv_logits, dim=1)
        adv_loss = loss_func(logits.float(), adv_logits)
        return adv_loss

    def adv_loss_similarity(self,
                            input_ids_1, attention_mask_1,
                            input_ids_2, attention_mask_2):
        input_ids = torch.cat([input_ids_1, input_ids_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)
        embed = self.forward(input_ids, attention_mask)
        logits = self.similarity(embed)

        noise = generate_noise(embed, epsilon=self.noise_var)
        for step in range(self.iter_var):
            adv_logits = self.paraphrase(embed + noise)
            adv_loss = F.mse_loss(adv_logits, logits.float(), reduction='sum')
            (delta_grad,) = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return 0
            # eff_delta_grad = delta_grad * self.lr
            delta_grad = noise + delta_grad * self.lr
            noise = self._norm_grad(delta_grad)
            noise = noise.detach()
            noise.requires_grad_()
        adv_logits = self.similarity(embed + noise)
        adv_loss = F.mse_loss(logits.float(), adv_logits, reduction='sum')
        return adv_loss

    def breg_loss_sentiment(self, curr_output: torch.Tensor, prev_output: torch.Tensor):
        curr_output = F.softmax(curr_output, dim=1)
        prev_output = F.softmax(prev_output, dim=1)
        return F.cross_entropy(curr_output, prev_output, reduction='sum')
    
    def breg_loss_paraphrase(self, curr_output: torch.Tensor, prev_output: torch.Tensor):
        curr_output = F.softmax(curr_output, dim=1)
        prev_output = F.softmax(prev_output, dim=1)
        return nn.BCEWithLogitsLoss(reduction='sum')(curr_output, prev_output)
    
    def breg_loss_similarity(self, curr_output: torch.Tensor, prev_output: torch.Tensor):
        curr_output = F.softmax(curr_output, dim=1)
        prev_output = F.softmax(prev_output, dim=1)
        return F.mse_loss(curr_output, prev_output, reduction='sum')
    
    def _norm_grad(self, grad):
        return grad / (torch.norm(grad, dim=-1, keepdim=True) + self.noise_var)




def generate_noise(embed, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    # to enable breg loss
    prev_prediction = {}
    
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train,
                                                                                      args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev,
                                                                                args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)
    # model_t for theta_tilt, load new parameters to this model
    model_t = MultitaskBERT(config)
    model_t = model_t.to(device)

    lr = args.lr
    # original AdamW
    optimizer = AdamW(model.parameters(), lr=lr)
    # AdamW with Bregman optimization
    # optimizer = AdamWBreg(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Define the momentum parameter
    beta = 0.9

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        # for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):

        zip_dataloader = zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader)

        for batch in tqdm(zip_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            sst_b_ids, sst_b_mask, sst_b_labels = (batch[0]['token_ids'],
                                                   batch[0]['attention_mask'], batch[0]['labels'])
            para_b_ids_1, para_b_ids_2, para_b_mask_1, para_b_mask_2, para_b_labels = (batch[1]['token_ids_1'],
            batch[1]['token_ids_2'], batch[1]['attention_mask_1'], batch[1]['attention_mask_2'], batch[1]['labels'])
            sts_b_ids_1, sts_b_ids_2, sts_b_mask_1, sts_b_mask_2, sts_b_labels = (batch[2]['token_ids_1'],
            batch[2]['token_ids_2'], batch[2]['attention_mask_1'], batch[2]['attention_mask_2'], batch[2]['labels'])

            sst_b_ids = sst_b_ids.to(device)
            sst_b_mask = sst_b_mask.to(device)
            sst_b_labels = sst_b_labels.to(device)
            para_b_ids_1 = para_b_ids_1.to(device)
            para_b_ids_2 = para_b_ids_2.to(device)
            para_b_mask_1 = para_b_mask_1.to(device)
            para_b_mask_2 = para_b_mask_2.to(device)
            para_b_labels = para_b_labels.to(device)
            sts_b_ids_1 = sts_b_ids_1.to(device)
            sts_b_ids_2 = sts_b_ids_2.to(device)
            sts_b_mask_1 = sts_b_mask_1.to(device)
            sts_b_mask_2 = sts_b_mask_2.to(device)
            sts_b_labels = sts_b_labels.to(device)

            optimizer.zero_grad()
            sst_logits = model.predict_sentiment(sst_b_ids, sst_b_mask)
            sst_loss = F.cross_entropy(sst_logits, sst_b_labels.view(-1), reduction='sum') / args.batch_size
            # sst_adv_loss = model.adv_loss_sentiment(sst_b_ids, sst_b_mask) / args.batch_size
            # bregman
            sst_breg_loss = model.breg_loss_sentiment(sst_logits, model_t.predict_sentiment(sst_b_ids, sst_b_mask))


            # print(f"sst_loss: {sst_loss: .3f}, sst_adv_loss: {sst_adv_loss: .3f}")
            # sst_loss = sst_loss + sst_adv_loss * model.lbda + sst_breg_loss * model.mu
            sst_loss = sst_loss + sst_breg_loss * model.mu


            para_logits = model.predict_paraphrase(para_b_ids_1, para_b_mask_1, para_b_ids_2, para_b_mask_2)
            logits_loss_function = nn.BCEWithLogitsLoss(reduction='sum')
            para_loss = logits_loss_function(para_logits.view(-1), para_b_labels.float()) / args.batch_size
            # para_adv_loss = model.adv_loss_paraphrase(para_b_ids_1, para_b_mask_1,
            #                                           para_b_ids_2, para_b_mask_2) / args.batch_size
            # print(f"p_loss: {para_loss: .3f}, p_adv_loss: {para_adv_loss: .3f}")

            # bregman
            para_breg_loss = model.breg_loss_paraphrase(para_logits, model_t.predict_paraphrase(para_b_ids_1, para_b_mask_1, para_b_ids_2, para_b_mask_2))
            # para_loss = para_loss + para_adv_loss * model.lbda + para_breg_loss * model.mu
            para_loss = para_loss + para_breg_loss * model.mu

            sts_logits = model.predict_similarity(sts_b_ids_1, sts_b_mask_1, sts_b_ids_2, sts_b_mask_2)
            sts_loss = F.mse_loss(sts_logits.view(-1), sts_b_labels.float(), reduction='sum') / args.batch_size
            # sts_adv_loss = model.adv_loss_similarity(sts_b_ids_1, sts_b_mask_1,
            #                                          sts_b_ids_2, sts_b_mask_2) / args.batch_size
            # print(f"sts_loss: {sts_loss: .3f}, sts_adv_loss: {sts_adv_loss: .3f}")
            
            # bregman
            sts_breg_loss = model.breg_loss_similarity(sts_logits, model_t.predict_similarity(sts_b_ids_1, sts_b_mask_1, sts_b_ids_2, sts_b_mask_2))
            # sts_loss = sts_loss + sts_adv_loss * model.lbda + sts_breg_loss * model.mu
            sts_loss = sts_loss + sts_breg_loss * model.mu

            loss = (sst_loss + para_loss + sts_loss) / 3

            # backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        

        # update: theta_tilt_t = (1 - beta) theta_t + beta * theta_tilt_t-1
        param_dict = {}
        for (n, p), (_, pt) in zip(model.named_parameters(), model_t.named_parameters()):
            param_dict[n] = (1 - model.beta) * p + model.beta * pt
        param_dict['bert.position_ids'] = model.state_dict()['bert.position_ids']
        model_t.load_state_dict(param_dict)

        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        para_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _ = \
            model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _ = \
            model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        train_acc = (para_train_acc + sst_train_acc + sts_train_corr) / 3

        dev_acc = (para_dev_acc + sst_dev_acc + sts_dev_corr) / 3

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
