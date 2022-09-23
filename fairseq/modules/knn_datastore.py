import torch
import faiss
import numpy as np
from torch_scatter import scatter
import time
import math
import faiss.contrib.torch_utils
import pickle

from threading import Thread


class KNN_Dstore(object):

    def __init__(self, args, trg_vocab_size):

        self.half = args.decoder.fp16
        self.dimension = args.decoder.embed_dim
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.use_gpu_to_search = args.use_gpu_to_search
        self.vocab_size = trg_vocab_size

        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        # set lambda
        self.set_lambda(args)

        # set temperature
        self.temperature_type = args.knn_temperature_type
        if self.temperature_type == 'fix':
            self.temperature = args.knn_temperature_value
        elif self.temperature_type == 'trainable':
            self.temperature = None
        else:
            self.temperature = None

        self.k = args.k
        
        self.dstore_2 = args.dstore_2
        self.dstore_3 = args.dstore_3
        if args.dstore_2:
            self.k_2 = args.k_2
            self.dstore_size_2 = args.dstore_size_2
            self.lambda_value_2 = args.knn_lambda_value_2
            self.temperature_2 = args.knn_temperature_value_2

        if args.dstore_3:
            self.k_3 = args.k_3
            self.dstore_size_3 = args.dstore_size_3
            self.lambda_value_3 = args.knn_lambda_value_3
            self.temperature_3 = args.knn_temperature_value_3

        self.mask_for_label_count = self.generate_label_count_mask(args.k)

        self.index, self.index_2, self.index_3 = self.setup_faiss(args)

    def generate_neighbor_mask(self, max_k):

        # [1, 1000, 1000]
        # [1, 1,    1000]
        # [1, 1,    1   ]
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1

        # we only select 2's power here
        # [1 - 1, 2 - 1, 4 - 1, 8 - 1, ...]
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(self.max_k, 2)) + 1)])
        k_mask = k_mask[power_index]

        k_mask.requires_grad = False
        if torch.cuda.is_available():
            k_mask = k_mask.cuda()

        return k_mask

    def generate_label_count_mask(self, max_k):

        # [0, 1, 1]
        # [0, 0, 1]
        # [0, 0, 0]
        mask_for_label_count = torch.empty((max_k, max_k)).fill_(1)
        mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()

        if torch.cuda.is_available():
            mask_for_label_count = mask_for_label_count.cuda()

        mask_for_label_count.requires_grad = False

        return mask_for_label_count

    def get_label_count_segment(self, tgt_idx: torch.Tensor, relative=False):  # [B, S, K]
        """
        This function return the label counts for different range of k nearest neighbor
        [[0:0], [0:1], [0:2], ..., [0:K-1]]

        """

        B, S, K = tgt_idx.size()

        expand_tgt_idx = tgt_idx.unsqueeze(-2).expand(B, S, K, K)
        expand_tgt_idx = expand_tgt_idx.masked_fill(self.mask_for_label_count, value=-1)

        labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
        retrieve_label_counts[:, :, :-1] -= 1

        # if we want relative label count, i.e [1, 2, 3, 3, 4] -> [1, 1, 1, 0, 1]
        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]

        return retrieve_label_counts

    def get_label_count(self, tgt_idx: torch.Tensor):
        """
        This only return total label count for all neighbors
        """
        tgt_sorted, _ = tgt_idx.sort(dim=-1)
        tgt_sorted[:, :, 1:] *= ((tgt_sorted[:, :, 1:] - tgt_sorted[:, :, :-1]) != 0).long()
        retrieve_label_counts = tgt_sorted.ne(0).sum(-1).unsqueeze(-1)  # [B, S, 1]

        return retrieve_label_counts

    def set_lambda(self, args):

        if not hasattr(args, 'knn_lambda_type'):
            return

        self.lambda_type = args.knn_lambda_type

        if self.lambda_type == 'fix':
            self.lambda_value = args.knn_lambda_value

        if self.lambda_type == 'trainable':
            self.lambda_value = None  # not generate lambda value in this class

    def get_faiss_centroids(self):
        return self.index.quantizer.reconstruct_n(0, self.index.nlist)

    def get_lambda(self, step=None, distance=None):

        if self.lambda_type == 'fix':
            if self.dstore_2 and self.dstore_3:
                return self.lambda_value, self.lambda_value_2, self.lambda_value_3 
            elif self.dstore_2:
                return self.lambda_value, self.lambda_value_2, None
            else:
                return self.lambda_value, None, None

        elif self.lambda_type == 'trainable':

            return None

    def get_temperature(self):

        if self.temperature_type == 'fix':
            if self.dstore_2 and self.dstore_3:
                return self.temperature, self.temperature_2, self.temperature_3
            elif self.dstore_2:
                return self.temperature, self.temperature_2, None
            else:
                return self.temperature, None, None
        else:
            return None

    def setup_faiss(self, args):

        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.dstore_filename + 'knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
        if self.dstore_2:
            index_2 = faiss.read_index(args.dstore_filename_2 + 'knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
        if self.dstore_3:
            index_3 = faiss.read_index(args.dstore_filename_3 + 'knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)

        if self.use_gpu_to_search:
            print('put index from cpu to gpu')
            res = faiss.StandardGpuResources()
            self.res = res
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

            if self.dstore_2:
                print('put index 2 from cpu to gpu')
                index_2 = faiss.index_cpu_to_gpu(res, 0, index_2, co)

            if self.dstore_3:
                print('put index 3 from cpu to gpu')
                index_3 = faiss.index_cpu_to_gpu(res, 0, index_3, co)


        print('Reading datastore took {} s'.format(time.time() - start))
        print('the datastore is {}, size is {}, and dim is {} '.format(args.dstore_filename, self.dstore_size, self.dimension))
        if self.dstore_2:
            print('the datastore 2 is {}, size is {}, and dim is {} '.format(args.dstore_filename_2, self.dstore_size_2, self.dimension))
        if self.dstore_3:
            print('the datastore 3 is {}, size is {}, and dim is {} '.format(args.dstore_filename_3, self.dstore_size_3, self.dimension))

        index.nprobe = args.probe
        if self.dstore_2:
            index_2.nprobe = args.probe
        if self.dstore_3:
            index_3.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + 'keys.npy', dtype=np.float16, mode='r',shape=(self.dstore_size, self.dimension))
                if self.dstore_2:
                    self.keys_2 = np.memmap(args.dstore_filename_2 + 'keys.npy', dtype=np.float16, mode='r',shape=(self.dstore_size_2, self.dimension))
                if self.dstore_3:
                    self.keys_3 = np.memmap(args.dstore_filename_3 + 'keys.npy', dtype=np.float16, mode='r',shape=(self.dstore_size_3, self.dimension))

            self.vals = np.memmap(args.dstore_filename + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size, 1))
            if self.dstore_2:
                self.vals_2 = np.memmap(args.dstore_filename_2 + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size_2, 1))
            if self.dstore_3:
                self.vals_3 = np.memmap(args.dstore_filename_3 + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size_3, 1))


        else:
            print('Keys are fp32 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + 'keys.npy', dtype=np.float32, mode='r',shape=(self.dstore_size, self.dimension))
                if self.dstore_2:
                    self.keys_2 = np.memmap(args.dstore_filename_2 + 'keys.npy', dtype=np.float32, mode='r',shape=(self.dstore_size_2, self.dimension))
                if self.dstore_3:
                    self.keys_3 = np.memmap(args.dstore_filename_3 + 'keys.npy', dtype=np.float32, mode='r',shape=(self.dstore_size_3, self.dimension))

            self.vals = np.memmap(args.dstore_filename + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size, 1))
            if self.dstore_2:
                self.vals_2 = np.memmap(args.dstore_filename_2 + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size_2, 1))
            if self.dstore_3:
                self.vals_3 = np.memmap(args.dstore_filename_3 + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size_3, 1))


        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '/keys.npy',
                                                  dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

                if self.dstore_2:
                    del self.keys_2
                    self.keys_from_memmap_2 = np.memmap(args.dstore_filename_2 + '/keys.npy',
                                                      dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                      shape=(self.dstore_size_2, self.dimension))
                    self.keys_2 = np.zeros((self.dstore_size_2, self.dimension),
                                         dtype=np.float16 if args.dstore_fp16 else np.float32)
                    self.keys_2 = self.keys_from_memmap_2[:]
                    self.keys_2 = self.keys_2.astype(np.float16 if args.dstore_fp16 else np.float32)

                if self.dstore_3:
                    del self.keys_3
                    self.keys_from_memmap_3 = np.memmap(args.dstore_filename_3 + '/keys.npy',
                                                      dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                      shape=(self.dstore_size_3, self.dimension))
                    self.keys_3 = np.zeros((self.dstore_size_3, self.dimension),
                                         dtype=np.float16 if args.dstore_fp16 else np.float32)
                    self.keys_3 = self.keys_from_memmap_3[:]
                    self.keys_3 = self.keys_3.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + 'vals.npy',
                                              dtype=np.int, mode='r',
                                              shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int)

            if self.use_gpu_to_search:
                self.vals = torch.from_numpy(self.vals)
                if torch.cuda.is_available():
                    print('put vals to gpu')
                    self.vals = self.vals.cuda()

            if self.dstore_2:
                del self.vals_2
                self.vals_from_memmap_2 = np.memmap(args.dstore_filename_2 + 'vals.npy',
                                                  dtype=np.int, mode='r',
                                                  shape=(self.dstore_size_2, 1))
                self.vals_2 = np.zeros((self.dstore_size_2, 1), dtype=np.int)
                self.vals_2 = self.vals_from_memmap_2[:]
                self.vals_2 = self.vals_2.astype(np.int)

                if self.use_gpu_to_search:
                    self.vals_2 = torch.from_numpy(self.vals_2)
                    if torch.cuda.is_available():
                        print('put vals 2 to gpu')
                        self.vals_2 = self.vals_2.cuda()

            if self.dstore_3:
                del self.vals_3
                self.vals_from_memmap_3 = np.memmap(args.dstore_filename_3 + 'vals.npy',
                                                  dtype=np.int, mode='r',
                                                  shape=(self.dstore_size_3, 1))
                self.vals_3 = np.zeros((self.dstore_size_3, 1), dtype=np.int)
                self.vals_3 = self.vals_from_memmap_3[:]
                self.vals_3 = self.vals_3.astype(np.int)

                if self.use_gpu_to_search:
                    self.vals_3 = torch.from_numpy(self.vals_3)
                    if torch.cuda.is_available():
                        print('put vals 3 to gpu')
                        self.vals_3 = self.vals_3.cuda()


            print('Loading to memory took {} s'.format(time.time() - start))
        if self.dstore_2 and self.dstore_3:
            return index, index_2, index_3
        elif self.dstore_2:
            return index, index_2, None
        else:
            return index, None, None

    def dist_func(self, d, k, q, function=None):

        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                if torch.cuda.is_available():
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                else:
                    knns_vecs = torch.from_numpy(self.keys[k]).view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                return -1 * l2
            return d

        if function == 'dot':
            qsize = q.shape
            if torch.cuda.is_available():
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)
            else:
                return (torch.from_numpy(self.keys[k]) * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")


    def get_knns(self, queries):

        # move query to numpy, if faiss version < 1.6.5
        # numpy_queries = queries.detach().cpu().float().numpy()

        dists, knns = self.index.search(queries, self.k)
        if self.dstore_2:
            dists_2, knns_2 = self.index_2.search(queries, self.k_2)
        if self.dstore_3:
            dists_3, knns_3 = self.index_3.search(queries, self.k_3)

        if self.dstore_2 and self.dstore_3:
            return dists, knns, dists_2, knns_2, dists_3, knns_3
        elif self.dstore_2:
            return dists, knns, dists_2, knns_2
        else:
            return dists, knns


    def retrieve(self, queries):

        # queries  are [Batch, seq len, Hid Size]

        # retrieve
        bsz = queries.size(0)
        seq_len = queries.size(1)
        
        if self.dstore_2 and self.dstore_3:
            dists, knns, dists_2, knns_2, dists_3, knns_3 = self.get_knns(queries.contiguous().view(-1, queries.size(-1)).cpu())  # [Batch * seq len, K]
        elif self.dstore_2:
            dists, knns, dists_2, knns_2 = self.get_knns(queries.contiguous().view(-1, queries.size(-1)).cpu())  # [Batch * seq len, K]
        else:
            dists, knns = self.get_knns(queries.contiguous().view(-1, queries.size(-1)).cpu())  # [Batch * seq len, K]
    
        if not self.use_gpu_to_search:
            tgt_idx = torch.from_numpy(self.vals[knns]).to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
            if self.dstore_2:
                tgt_idx_2 = torch.from_numpy(self.vals_2[knns_2]).to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
            if self.dstore_3:
                tgt_idx_3 = torch.from_numpy(self.vals_3[knns_3]).to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
        else:
            tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)
            if self.dstore_2:
                tgt_idx_2 = self.vals_2[knns_2].to(queries.device).squeeze(-1)
            if self.dstore_3:
                tgt_idx_3 = self.vals_3[knns_3].to(queries.device).squeeze(-1)
        
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)  # [B, S, K]
        if self.dstore_2:
            tgt_idx_2 = tgt_idx_2.view(bsz, seq_len, -1)  # [B, S, K]
        if self.dstore_3:
            tgt_idx_3 = tgt_idx_3.view(bsz, seq_len, -1)  # [B, S, K]

        if torch.cuda.is_available():
            dists = dists.view(bsz, seq_len, -1).cuda()  # [Batch, Seq len, k]
            knns = knns.view(bsz, seq_len, -1).cuda()
            if self.dstore_2:
                dists_2 = dists_2.view(bsz, seq_len, -1).cuda()  # [Batch, Seq len, k]
                knns_2 = knns_2.view(bsz, seq_len, -1).cuda()
            if self.dstore_3:
                dists_3 = dists_3.view(bsz, seq_len, -1).cuda()  # [Batch, Seq len, k]
                knns_3 = knns_3.view(bsz, seq_len, -1).cuda()
        else:
            dists = dists.view(bsz, seq_len, -1)  # [Batch, Seq len, k]
            knns = knns.view(bsz, seq_len, -1)
            if self.dstore_2:
                dists_2 = dists_2.view(bsz, seq_len, -1)  # [Batch, Seq len, k]
                knns_2 = knns_2.view(bsz, seq_len, -1)
            if self.dstore_3:
                dists_3 = dists_3.view(bsz, seq_len, -1)  # [Batch, Seq len, k]
                knns_3 = knns_3.view(bsz, seq_len, -1)

        if self.dstore_2 and self.dstore_3:
            return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx,
                    'distance_2': dists_2, 'knn_index_2': knns_2, 'tgt_index_2': tgt_idx_2,
                    'distance_3': dists_3, 'knn_index_3': knns_3, 'tgt_index_3': tgt_idx_3}
        elif self.dstore_2:
            return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx,
                    'distance_2': dists_2, 'knn_index_2': knns_2, 'tgt_index_2': tgt_idx_2,
                    'distance_3': None, 'knn_index_3': None, 'tgt_index_3': None}
        else:
            return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx,
                    'distance_2': None, 'knn_index_2': None, 'tgt_index_2': None,
                    'distance_3': None, 'knn_index_3': None, 'tgt_index_3': None}

    def calculate_knn_prob(self,
                           knn_index: torch.Tensor,  # [B, S, K]
                           tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor,  # [B, S, K]
                           queries: torch.Tensor,  # [B, S, H]
                           temperature: torch.Tensor,  # [B, S, 1]
                           ):

        bsz = queries.size(0)
        seq_len = queries.size(1)

        # update the dist and compute each neighbor weight, neg distance
        re_compute_dists = self.dist_func(distance, knn_index, queries, function=self.sim_func)  # [B, S, K]

        scaled_dists = re_compute_dists / temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        # set the target index for each neighbor
        knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        # implemented with pytorch_scatter
        scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]
        

        return {'prob': prob}

    def update_get_knn_seq_prob(self, queries):

        knn_search_result = self.retrieve(queries)

        if self.temperature_type == 'fix':
            final_result = self.calculate_knn_prob(knn_index=knn_search_result['knn_index'],
                                                   tgt_index=knn_search_result['tgt_index'],
                                                   distance=knn_search_result['distance'],
                                                   queries=queries,
                                                   temperature=self.temperature)
            return {'distance': knn_search_result['distance'],
                    'knn_index': knn_search_result['knn_index'],
                    'prob': final_result['prob'],}


if __name__ == "__main__":
    class ARGS:
        fp16 = False
        decoder_embed_dim = 1024
        k = 64
        dstore_size = 524400
        faiss_metric_type = 'do_not_recomp_l2'
        knn_sim_func = 'do_not_recomp_l2'
        dstore_fp16 = True
        knn_temperature = 1.0
        indexfile = ''
        dstore_filename = ''
        no_load_keys = False
        probe = 32
        move_dstore_to_mem = True
        use_gpu_to_search = True
        trg_vocab_size = 42024


    args = ARGS()
    knn_store = KNN_Dstore(args=args, trg_vocab_size=args.trg_vocab_size)

    query = torch.randn(32 * 4, 1024)
    print('query size is {}', query.size())
    # dist, knn_idx = knn_store.get_knns(query)
    # print(dist.shape)  # [10000, 64]
    # print(knn_idx.shape)  # [10000, 64]

    prob = knn_store.get_knn_prob(query)
    # print(prob.max(dim=-1)[0])
    # print(prob.max(dim=-1)[1])

    print('average time for retrieve neighbors, {} s'.format(knn_store.time_for_retrieve / knn_store.retrieve_count))
    print('average time for set the target prob for each neighbor'
          ' (need do scatter operation for (batch size * beam size * k, vocab size) tensor), {} s'
          .format(knn_store.time_for_setup_prob / knn_store.retrieve_count))
