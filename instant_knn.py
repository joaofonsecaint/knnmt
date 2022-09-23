import logging
import os
import sys
import numpy as np
import faiss
import time
import torch
import ctypes

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

        #either create new memmap with corrections or append to existing datastore 
        if args.append_ds:
            print('Saving fp32')
            updated_dstore_keys = np.memmap(args.update_dstore + '/keys.npy', dtype=np.float32, mode='r+',
                                    shape=(updated_dstore_size, args.decoder_embed_dim))
            updated_dstore_vals = np.memmap(args.update_dstore + '/vals.npy', dtype=np.int, mode='r+',
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

                #add new keys and vals to 'corrected' datastore
                #new keys
                updated_dstore_keys[(args.dstore_size+dstore_idx):(args.dstore_size+reduce_size + dstore_idx)] = features1.detach().cpu().numpy().astype(
                    np.float32) 
                #new vals
                updated_dstore_vals[(args.dstore_size+dstore_idx):(args.dstore_size+reduce_size + dstore_idx)] = target1.unsqueeze(-1).cpu().numpy().astype(np.int) 
                

                
                dstore_idx += reduce_size

                if dstore_idx >= args.dstore_size:
                    print('much more than dstore size break')
                    break

        # ADD CORRECTIONS TO FAISS INDEX

        res = faiss.StandardGpuResources()
        # to speed up access to np.memmap
        madvise = ctypes.CDLL("libc.so.6").madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int
        assert madvise(updated_dstore_keys.ctypes.data, updated_dstore_keys.size * updated_dstore_keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

        #new_keys=updated_dstore_keys[args.dstore_size:updated_dstore_size]
        #new_vals=updated_dstore_vals[args.dstore_size:updated_dstore_size]

        np.random.seed(args.seed)
        #random_sample = np.random.choice(np.arange(new_keys.shape[0]), size=[min(1000000, new_keys.shape[0])],replace=False)
        
        #READ INDEX
        print("READING INDEX")
        index=faiss.read_index(args.faiss_index)

        print("READ INDEX SUCCESSFULLY")

        print("ADDING TO INDEX")
        #index.add(new_keys[random_sample].astype(np.float32))

        start_time = time.time()
        start=args.dstore_size
        num_keys_to_add_at_a_time= 1000000
        while start < updated_dstore_size:
            end = min(updated_dstore_size, start + num_keys_to_add_at_a_time)
            to_add = updated_dstore_keys[start:end].copy()

            index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))  #add or add with ids

            start += num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                print('Added %d tokens so far' % start)
                print('Writing Index', start)
                faiss.write_index(index, args.update_dstore+'/knn_index')

        print("Adding total %d keys" % end)
        print('Adding took {} s'.format(time.time() - start_time))

        faiss.write_index(index, args.update_dstore+'/knn_index')



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
