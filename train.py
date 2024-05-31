"""
Train a diffusion model on images.
"""

import argparse
import json
import os

import opacus
import torch
from transformers import set_seed

from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from diffuseq.step_sample import create_named_schedule_sampler
from diffuseq.text_datasets import load_data_text
from diffuseq.utils import logger
from train_util import TrainLoop

os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    parser.add_argument('--private', type=bool, default=True, help='run fine-tuning with DP')
    parser.add_argument('--epsilon', type=int, default=1, help='if dp, how large should epsilon be?')
    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse

    return parser


def main():
    args = create_argparser().parse_args()
    # update args
    args.learning_steps = 50000
    args.save_interval = 100
    args.dataset = 'qqp'
    args.data_dir = 'datasets/QQP'
    args.checkpoint_path = 'diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_test_ori20221113-20:27:29'
    args.config_name = 'bert-base-uncased'
    args.batch_size = 8


    set_seed(args.seed)
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        loaded_vocab=tokenizer,
        model_emb=model_weight  # use model's weights as init
    )


    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight  # using the same embedding wight with tranining data
    )

    print('#' * 30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("### Training...")

    # make private
    if args.private:
        privacy_engine = opacus.PrivacyEngine()
        model, opt, data = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=opt,
            data_loader=data,
            target_epsilon=args.epsilon,
            target_delta=(len(data.dataset) * 10) ** -1,
            epochs=1,
            max_grad_norm=1.0
        )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        opt=opt,
        private=args.private
    ).run_loop()


if __name__ == "__main__":
    main()
