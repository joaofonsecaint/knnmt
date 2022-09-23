import logging
import os
import sys
from itertools import chain
import argparse
import numpy as np
import faiss
import time
import torch
from tqdm import tqdm

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")



def main(args, override_args=None):
    utils.import_user_module(args)

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task([args.path],arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),)
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # --- check existent saved data store
    if args.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r',
                                shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r',
                                shape=(args.dstore_size, 1))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='r',
                                shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r',
                                shape=(args.dstore_size, 1))


    dstore_idx = 0
    dstore_idx2 = 0
    data_idx = 1
    for subset in args.valid_subset.split(","):
        try:
            model_args.dataset.required_seq_len_multiple = 1
            model_args.task.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=data_idx)
            data_idx = data_idx + 1
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        #updated mem map for corrections and previous content
        corrections_to_add=args.dstore_size_2
        updated_dstore_size=args.dstore_size+corrections_to_add

        #either create new datadtore with corrections or append to existing datastore 
        if args.append_ds:
            if args.dstore_fp16:
                print('Saving fp16')
                updated_dstore_keys = np.memmap(args.update_dstore + '/keys.npy', dtype=np.float16, mode='r+',
                                        shape=(updated_dstore_size, args.decoder_embed_dim))
                updated_dstore_vals = np.memmap(args.update_dstore + '/vals.npy', dtype=np.int, mode='r+',
                                        shape=(updated_dstore_size, 1))
            else:
                print('Saving fp32')
                updated_dstore_keys = np.memmap(args.update_dstore + '/keys.npy', dtype=np.float32, mode='r+',
                                        shape=(updated_dstore_size, args.decoder_embed_dim))
                updated_dstore_vals = np.memmap(args.update_dstore + '/vals.npy', dtype=np.int, mode='r+',
                                        shape=(updated_dstore_size, 1))
        else:
            if args.dstore_fp16:
                print('Saving fp16')
                updated_dstore_keys = np.memmap(args.update_dstore + '/keys.npy', dtype=np.float16, mode='w+',
                                        shape=(updated_dstore_size, args.decoder_embed_dim))
                updated_dstore_vals = np.memmap(args.update_dstore + '/vals.npy', dtype=np.int, mode='w+',
                                        shape=(updated_dstore_size, 1))
            else:
                print('Saving fp32')
                updated_dstore_keys = np.memmap(args.update_dstore + '/keys.npy', dtype=np.float32, mode='w+',
                                        shape=(updated_dstore_size, args.decoder_embed_dim))
                updated_dstore_vals = np.memmap(args.update_dstore + '/vals.npy', dtype=np.int, mode='w+',
                                        shape=(updated_dstore_size, 1))
                updated_dstore_keys[0:args.dstore_size] = dstore_keys[:] #old keys
                updated_dstore_vals[0:args.dstore_size] = dstore_vals[:] #old vals

        with open(args.update_dstore + "/ds_size",'w') as f:
            f.write(str(len(updated_dstore_vals)))


        # Create Second datastore with corrections only
        if args.num_datastores==1:
            with open("second_datastore/ds_size",'w') as f:
                f.write("/dev/null")           
        elif args.num_datastores==2:
            print('CREATING SECOND DATASTORE')
            second_dstore_keys = np.memmap('second_datastore/keys.npy', dtype=np.float32, mode='w+',
                                    shape=(corrections_to_add, args.decoder_embed_dim))
            second_dstore_vals = np.memmap('second_datastore/vals.npy', dtype=np.int, mode='w+',
                                    shape=(corrections_to_add, 1)) 
            with open("second_datastore/ds_size",'w') as f:
                f.write(str(len(second_dstore_keys)))
    
        # Initialize data iterator
        itr = task.get_batch_iterator(dataset=dataset, max_tokens=args.max_tokens, max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(task.max_positions(), *[m.max_positions() for m in models],),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple, seed=args.seed,
            num_shards=args.distributed_world_size, shard_id=args.distributed_rank, num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,).next_epoch_itr(False)
        
        progress = progress_bar.progress_bar(itr, log_format=args.log_format, log_interval=args.log_interval,
           prefix=f"valid on '{subset}' subset", default_log_format=("tqdm" if not args.no_progress_bar else "simple"),)

        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                features = task.forward_and_get_hidden_state_step(sample, model)  # [B, T, H]
                target = sample['target']  # [B, T]
                
                # get useful parameters
                batch_size = target.size(0)
                seq_len = target.size(1)
                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]

                # remove the pad tokens and related hidden states
                target = target.view(batch_size * seq_len)
                target_mask = target_mask.view(batch_size * seq_len)

                non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                target = target.index_select(dim=0, index=non_pad_index)  # [n_count]

                features = features.contiguous().view(batch_size * seq_len, -1)
                features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]

                # save to the dstore
                current_batch_count = target.size(0)
                if (args.dstore_size + dstore_idx) + current_batch_count > updated_dstore_size:
                    reduce_size = updated_dstore_size - (dstore_idx+args.dstore_size)
                    features1 = features[:reduce_size]
                    target1 = target[:reduce_size]
                    #if args.save_plain_text:
                    #    src_tokens = src_tokens[:reduce_size, :]
                else:
                    reduce_size = current_batch_count
                    features1=features
                    target1=target

                #add new and old keys and vals to 'corrected' datastore
                if args.dstore_fp16:
                    updated_dstore_keys[(args.dstore_size+dstore_idx):(args.dstore_size+reduce_size + dstore_idx)] = features1.detach().cpu().numpy().astype(
                        np.float16)

                    updated_dstore_vals[(args.dstore_size+dstore_idx):(args.dstore_size+reduce_size + dstore_idx)] = target1.unsqueeze(-1).cpu().numpy().astype(np.int)
                else:
                    updated_dstore_keys[(args.dstore_size+dstore_idx):(args.dstore_size+reduce_size + dstore_idx)] = features1.detach().cpu().numpy().astype(
                        np.float32) #new keys

                    updated_dstore_vals[(args.dstore_size+dstore_idx):(args.dstore_size+reduce_size + dstore_idx)] = target1.unsqueeze(-1).cpu().numpy().astype(np.int) #new vals


                if args.num_datastores==2:
                    if dstore_idx2 + current_batch_count > len(second_dstore_keys):
                        reduce_size2 = len(second_dstore_keys) - dstore_idx2
                        features2 = features[:reduce_size2]
                        target2 = target[:reduce_size2]
                    else:
                        reduce_size2 = current_batch_count
                        features2=features
                        target2=target
                
                #Second datastore with corrections only
                if args.num_datastores==2:
                    second_dstore_keys[dstore_idx2:reduce_size2 + dstore_idx2] = features2.detach().cpu().numpy().astype(
                        np.float32)
                    second_dstore_vals[dstore_idx2:reduce_size2 + dstore_idx2] = target2.unsqueeze(-1).cpu().numpy().astype(np.int)


                dstore_idx += reduce_size
                if args.num_datastores==2:
                    dstore_idx2 += reduce_size2

                if dstore_idx >= args.dstore_size:
                    print('much more than dstore size break')
                    break

def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    #distributed_utils.call_main(args, main, override_args=override_args)
    main(args, override_args=override_args)

if __name__ == "__main__":
    cli_main()
