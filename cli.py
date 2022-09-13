import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default="/home/gamir/liat/UnberteseNew/data",
                        type=str,
                        required=False)
    parser.add_argument("--train_data_path",
                        default="/home/gamir/liat/UnberteseNew/data/lpqa_correct_results.json",
                        type=str,
                        required=False)
    parser.add_argument("--test_data_path",
                        default="/home/gamir/liat/UnberteseNew/data/lama_correct_results.json",
                        type=str,
                        required=False)
    parser.add_argument("--cache_dir",
                        default="/home/gamir/liat/Runs/cache_dir",
                        type=str,
                        required=False)
    parser.add_argument("--pretrained_model_path",
                        default="/home/gamir/liat/bert-base-uncased-identity-mse-sum-checkpoint-51031",
                        type=str,
                        required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese")
    parser.add_argument("--mlm_bert_model_name",
                        default="bert-base-uncased",
                        type=str,
                        required=False,
                        help="Bert pre-trained model selected in the list: lower_bert-base-uncased, "
                             "bert-large-uncased, lower_bert-base-cased, lower_bert-base-multilingual, lower_bert-base-chinese")
    parser.add_argument("--paraphrase_model_name",
                        default="princeton-nlp/sup-simcse-bert-base-uncased",
                        type=str,
                        required=False)
    parser.add_argument("--paraphrase_classifer_model_name",
                        default="Prompsit/paraphrase-bert-en",
                        type=str,
                        required=False)
    parser.add_argument("--output_dir",
                        default="/specific/netapp5_2/gamir/liat/Runs",
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dont_save",
                        default=False,
                        action='store_true')

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=20,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--is_debug",
                        default=False,
                        action='store_true')
    parser.add_argument("--use_wandb",
                        default=False,
                        action='store_true')
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='weight decay')
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps",
                        default=100,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--mask_loss_weight',
                        type=float,
                        default=0.5)
    parser.add_argument('--tokens_dist_loss_weight',
                        type=float,
                        default=0.3)
    parser.add_argument('--paraphrase_loss_weight',
                        type=float,
                        default=220)
    parser.add_argument('--label_loss_weight',
                        type=float,
                        default=0.2)

    parser.add_argument(
        "--checkpoint_file_name",
        type=str,
        default="checkpoints/unbertese",
        help="Checkpoints file prefix",
    )

    return parser.parse_args()

    
def parse_adv_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path",
                        default="/home/gamir/liat/UnberteseNew/data/rewrite_bad_entities_failed_only.json",
                        type=str,
                        required=False,)
    parser.add_argument("--test_data_path",
                        default="/home/gamir/liat/UnberteseNew/data/unified_all_lpqa_failed_entities.json",
                        type=str,
                        required=False,)
    parser.add_argument("--cache_dir",
                        default="/home/gamir/liat/Runs/cache_dir",
                        type=str,
                        required=False)
    parser.add_argument("--mlm_bert_model_name",
                        default="bert-base-uncased",
                        type=str,
                        required=False,
                        help="Bert pre-trained model selected in the list: lower_bert-base-uncased, "
                             "bert-large-uncased, lower_bert-base-cased, lower_bert-base-multilingual, lower_bert-base-chinese")
    parser.add_argument("--output_dir",
                        default="/specific/netapp5_2/gamir/liat/adv_Runs",
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dont_save",
                        default=False,
                        action='store_true')

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=20,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--is_debug",
                        default=False,
                        action='store_true')
    parser.add_argument("--use_wandb",
                        default=False,
                        action='store_true')
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='weight decay')
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps",
                        default=100,
                        type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--checkpoint_file_name",
        type=str,
        default="checkpoints/adv_unbertese",
        help="Checkpoints file prefix",
    )

    return parser.parse_args()

