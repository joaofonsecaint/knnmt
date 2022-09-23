
# kNN-MT
Implementation of the kNN-MT model.
The implementation is built upon [fairseq](https://github.com/pytorch/fairseq) and inspired on the [Adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt) implementation.

### Requirements

To run the code first, you need to install this repo as:

```
pip install --editable .
```

You  also need to install faiss. You can do it as:
```
conda install -c pytorch faiss-gpu
```
And  pytorch-scatter which you can install as:
```
pip install torch-scatter==2.0.5
```

####  Download pre-trained model
The pre-trained model used can be downloaded [here](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md). (De->En Single Model)

#### Download and process data
You can download the multi-domains dataset from [here](https://github.com/roeeaharoni/unsupervised-domain-clusters).

Then, to pre-process the data you should run this script for each domain:
```
bash ./examples/translation/prepare-domadapt.sh <path/to/domain/dir> <path/to/moses/and/fastBPE/dirs/>
```
Then prepare the binary files as:
```
python fairseq_cli/preprocess.py --source-lang de --target-lang en --trainpref <path/to/domain/dir>/processed/train.bpe.filtered --validpref <path/to/domain/dir>/processed/dev.bpe --testpref <path/to/domain/dir>/processed/medical/test.bpe --destdir <path/to/domain/dir>/data-bin/ --srcdict <path/to/model/dir>/dict.de.txt --joined-dictionary
```

### Create datastore
To create a datastore run: (to know the datastore size, you can check the preprocess log on the data-bin folder)
```
python create_datastore.py <path/to/domain/dir>/data-bin/ --dataset-impl mmap --task translation     --valid-subset train --path <path/to/model>  --max-tokens 4096 --skip-invalid-size-inputs-valid-test --decoder-embed-dim 1024 --dstore-size <size of datastore> --dstore-mmap <path/to/save/datastore>
```

### Train datastore
To train the datastore run: (If not using PCA to reduce keys size, change to --PCA=0)
```
python3 train_datastore.py --dstore_mmap <path/to/datastore> --dstore_size <size of datastore> --faiss_index <path/to/save/faiss/index> --pca <PCA output dimension>
```


### Perform inference
To perform inference simply run:
```
python3 generate_knnmt.py <path/to/data/bin> --path <path/to/model> --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size <batch size> --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '<path/to/datasore>', 'dstore_size': <datastore size>, 'k': <number of neighbors to retrieve>, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': <lambda value>, 'knn_temperature_type': 'fix', 'knn_temperature_value': < retrieval softmax temperature>,}" 
```

To use multiple datastores (implementation only allows 3 but if needed is easy to increase) you just need to add the following arguments to the previous command (inside model-overrides). If using 2 datastores add:
```
'dstore_2': True, 'dstore_filename_2': '<path/to/datasore/2>', 'dstore_size_2': <datastore 2 size>, 'k_2': <number of neighbors to retrieve>, 'knn_lambda_value_2': <lambda value 2>, 'knn_temperature_value': <retrieval softmax temperature 2>
```
If using 3 datastores also add:
```
'dstore_3': True, 'dstore_filename_3': '<path/to/datasore/3>', 'dstore_size_3': <datastore 3 size>, 'k_3': <number of neighbors to retrieve>, 'knn_lambda_value_3': <lambda value 3>, 'knn_temperature_value': <retrieval softmax temperature 3>
```


### Make a Datastore Correction
Insert ALL your corrections in file corrections.txt