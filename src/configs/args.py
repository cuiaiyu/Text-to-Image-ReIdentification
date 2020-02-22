import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def load_arg_parser():
    parser = argparse.ArgumentParser(description='Aiyu ReID Training')
    # Experiment Miscs
    parser.add_argument('--note', 
                        default='', 
                        type=str,
                        help='note to add')
    
    parser.add_argument('--data_root', 
                        default='/shared/rsaas/aiyucui2/wider_person/', 
                        type=str,
                        help='dataroot')

    parser.add_argument('--output_path', 
                        default='outputs/reID/', 
                        type=str, metavar='PATH',
                        help='path to outputs')
    parser.add_argument('--model_path', 
                        default='checkpoints/reID/', 
                        type=str, metavar='PATH',
                        help='dir to save checkpoints')
    # data 
    parser.add_argument('--anno_path', 
                        default="wider/train/train_anns_train.json", 
                        type=str, metavar='PATH',
                        help='path to annotations')
    parser.add_argument('--img_dir', 
                        default="wider/train/img", 
                        type=str, metavar='PATH',
                        help='path to imgs')
    parser.add_argument('--mask_dir', 
                        default="wider/train/aiyu_binary_masks", 
                        type=str, metavar='PATH',
                        help='path to train masks')
    
    parser.add_argument('--val_anno_path', 
                        default="wider/val1/val1_anns.json", 
                        type=str, metavar='PATH',
                        help='path to annotations')
    parser.add_argument('--val_img_dir', 
                        default='wider/val1/img', 
                        type=str, metavar='PATH',
                        help='path to imgs')
    parser.add_argument('--val_mask_dir', 
                        default="wider/val1/aiyu_binary_masks", 
                        type=str, metavar='PATH',
                        help='path to val1 masks')
    parser.add_argument('--vocab_path', 
                        default='wider_graph/raw_vocab_th20.json', 
                        type=str, metavar='PATH',
                        help='path to pre-built vocab')
    parser.add_argument('--gt_file_fn', 
                        default='wider/val1/val_label.json', 
                        type=str,
                        help='file of gt - retrieval ground truth')
    parser.add_argument('--cheap_candidate_fn', 
                        default= 'wider_graph/summer_best_val1_top100.pkl', 
                        type=str,
                        help='file path to pre-computed candidates for val1')
    parser.add_argument('--load_ckpt_fn', 
                        default= '0', 
                        type=str,
                        help='continue training. 0 for fresh model.')
    
    

    # Models
    parser.add_argument('--load_model_path', default='starter_bert_resnet50_2048.pt', type=str,
                        help='0 indicating nothing')
    
    parser.add_argument('--np', default=False, type=str2bool,
                        help='use noun phrases or not?')
    parser.add_argument('--mask', default=False, type=str2bool,
                        help='use human parse segmask or not?')
    
    parser.add_argument('--img_num_cut', default=1, type=int,
                        help='how many cut on feature map (horizontal)?')
    parser.add_argument('--regional_embed_size', default=256, type=int,
                        help='regional_embed_size')

    parser.add_argument('--sent_token_length', 
                        default=40, type=int,
                        help='fixed length of tokenized sentences. (Run stats to decided, 40 by default for WIDER-person)')
    parser.add_argument('--np_token_length', 
                        default=6, type=int,
                        help='fixed length of tokenized NP. (Run stats to decided, 6 by default for WIDER-person)')
    parser.add_argument('--num_np_per_sent', 
                        default=6, type=int,
                        help='how many NPs per sent?. (Run stats to decided, 6 by default for WIDER-person)')
    
    
    parser.add_argument('--img_backbone_opt', 
                        default='resnet50', type=str,
                        choices=['resnet50','resnet18', 'resnet101'],
                        help='global encoder opt')
    parser.add_argument('--cap_embed_type', 
                        default='sent', type=str,
                        choices=['sent','word'],
                        help='global encoder opt')
    parser.add_argument('--cap_backbone_opt', 
                        default='bigru', type=str,
                        choices=['bigru'],
                        help='global encoder opt')
    
    parser.add_argument('--text_melt_layer', default=0, type=int, 
                        help='bert ft layer. 0 by default (layer residual layer, start from 11(0).')
    parser.add_argument('--image_melt_layer', default=1, type=int, 
                        help='resnet ft layer. 0 by default (layer residual layer, start from 8(0).')


    # Training Misc.
    parser.add_argument('--dist_fn_opt', 
                        default='euclidean', type=str,
                        choices=['euclidean','cosine'],
                        help='distance metrics opt')
    
    parser.add_argument('--num_gpus', 
                        default=1, type=int,
                        help='how much gpus to use?')
    parser.add_argument('--num_epochs_stage1', 
                        default=20, type=int,
                        help='as is')
    parser.add_argument('--num_epochs_stage2', 
                        default=60, type=int,
                        help='as is')
    parser.add_argument('--lr', default=2e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='SGD momentum')
    parser.add_argument('--weight_decay', default=0, type=float, 
                        help='SGD weight decay')
    parser.add_argument('--num_epochs', default=25, type=int, 
                        help='# epochs')
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='batch size')
    parser.add_argument('--step_size', default=15, type=int, 
                        help='lr step size (epoch)')
    

    # general misc.
    parser.add_argument('--cos_margin', default=0.5, type=float, 
                        help='# batch to log')
    parser.add_argument('--print_freq', default=50, type=int, 
                        help='# batch to log')
    parser.add_argument('--ckpt_freq', default=10, type=int, 
                        help='# epoch to save checkpoint')
    parser.add_argument('--embed_size', default=2048, type=int, 
                        help='size of embedding')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='# GPU used')

    # GPU
    parser.add_argument('--debug', default=True, type=str2bool, 
                        help='debug mode?')
    parser.add_argument('--cheap_eval', default=True, type=str2bool, 
                        help='run cheap cheap mode?')

    parser.add_argument('--num_workers', default=8, type=int, 
                        help='num workers of dataloader')

    # Mode
    parser.add_argument('--mode', default='train', type=str,choices=['train','val'],
                        help='# epoch to save checkpoint')

    parser.add_argument('--optimizer', default='Adam', type=str,choices=['SparseAdam','Adam'],
                        help='Opitimizer')
    return parser