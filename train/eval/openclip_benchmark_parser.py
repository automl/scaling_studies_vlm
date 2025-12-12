import argparse

from clip_benchmark.models import MODEL_TYPES


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+",
                             help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default="root", type=str,
                             help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--test_split', dest="split", action='store', type=str, default="test",
                             help="Dataset split to use")
    parser_eval.add_argument('--train_split', type=str, nargs="+", default="train", help="Dataset(s) train split names")
    mutually_exclusive = parser_eval.add_mutually_exclusive_group()
    mutually_exclusive.add_argument('--val_split', default=None, type=str, nargs="+",
                                    help="Dataset(s) validation split names. Mutually exclusive with val_proportion.")
    mutually_exclusive.add_argument('--val_proportion', default=None, type=float, nargs="+",
                                    help="what is the share of the train dataset will be used for validation part, if it doesn't predefined. Mutually exclusive with val_split")
    parser_eval.add_argument('--model', type=str, nargs="+", default=["ViT-B-32-quickgelu"],
                             help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--pretrained', type=str, nargs="+", default=["laion400m_e32"],
                             help="Model checkpoint name to use from OpenCLIP")
    parser_eval.add_argument('--pretrained_model', type=str, default="", nargs="+",
                             help="Pre-trained model(s) to use. Can be the full model name where `model` and `pretrained` are comma separated (e.g., --pretrained_model='ViT-B-32-quickgelu,laion400m_e32'), a model collection name ('openai' or 'openclip_base' or 'openclip_multilingual' or 'openclip_all'), or path of a text file where each line is a model fullname where model and pretrained are comma separated (e.g., ViT-B-32-quickgelu,laion400m_e32). --model and --pretrained are ignored if --pretrained_model is used.")
    parser_eval.add_argument('--task', type=str, default="auto",
                             choices=["zeroshot_classification", "zeroshot_retrieval", "linear_probe", "captioning",
                                      "image_caption_selection", "auto"],
                             help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True,
                             help="whether to use mixed precision")
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--recall_k', default=[5], type=int,
                             help="for retrieval, select the k for Recall@K metric. ", nargs="+", )
    parser_eval.add_argument('--fewshot_k', default=-1, type=int,
                             help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float,
                             help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true",
                             help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument("--distributed", action="store_true", help="evaluation in parallel")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--normalize', default=True, type=bool, help="features normalization")
    parser_eval.add_argument('--model_cache_dir', default=None, type=str,
                             help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str,
                             help="feature root folder where the features are stored.")
    parser_eval.add_argument('--annotation_file', default="", type=str,
                             help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser_eval.add_argument('--custom_classname_file', default=None, type=str,
                             help="use custom json file with classnames for each dataset, where keys are dataset names and values are list of classnames.")
    parser_eval.add_argument('--custom_template_file', default=None, type=str,
                             help="use custom json file with prompts for each dataset, where keys are dataset names and values are list of prompts. For instance, to use CuPL prompts, use --custom_template_file='cupl_prompts.json'")
    parser_eval.add_argument('--dump_classnames', default=False, action="store_true",
                             help="dump classnames to the results json file.")
    parser_eval.add_argument('--dump_templates', default=False, action="store_true",
                             help="dump templates to the results json file.")

    parser_eval.add_argument('--language', default="en", type=str, nargs="+",
                             help="language(s) of classname and prompts to use for zeroshot classification.")
    parser_eval.add_argument('--output', default="{dataset}_{pretrained}_{model}_{language}_{task}.json", type=str,
                             help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_{pretrained}_{model}_{language}_{task}.json'")
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false", help="suppress verbose messages")
    parser_eval.add_argument('--save_clf', default=None, type=str,
                             help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str,
                             help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true",
                             help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=MODEL_TYPES, help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str,
                             help="optional cache directory for webdataset only")
    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str, nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str, default="benchmark.csv", help="CSV output file")
    parser_build.set_defaults(which='build')

    return parser
