
import os
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

import pytorch_lightning as pl
import simple_parsing
import torch
import torch.optim

################################## Global parameters ##################################

@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    wandb_entity    : str          = "altegrad-gnn-link-prediction"         # name of the project
    debug           : bool         = False            # test code before running, if testing, no checkpoints are written
    wandb_project   : str          = f"test-altegrad"
    root_dir        : str          = os.getcwd()  # root_dir
    seed_everything : Optional[int]= None   # seed for the whole run
    tune_lr         : bool         = True  # tune the model on first run
    gpu             : int          = 1      # number or gpu
    max_epochs      : int          = 30    # maximum number of epochs
    weights_path    : str          = "weights"
    dev_run         : bool         = False
    train           : bool         = True
    best_model      : str          = "elated-aardvark-644" # then galant
    

@dataclass
class NetworkParams:
    network_name       : Optional[str] = "MLP"     # dataset, use <Dataset>Eval for FT
    weight_checkpoints : str           = ""
    artifact           : str           = ""
    vocab_size         : int           = 138499
    dropout         : float = 0.75
    normalization   : str   = 'BatchNorm1d'
    activation      : str   = 'GELU'
    input_size      : int   = 0    # dummy arg
    nb_authors      : int   = 149682
    emb_authors_dim : int   = 64

@dataclass
class OptimizerParams: 
    """Optimization parameters"""

    optimizer     : str   = "Adam"  # Optimizer default vit: AdamW, default resnet50: Adam
    lr            : float = 0.003     # learning rate,               default = 5e-4
    min_lr        : float = 5e-9     # min lr reached at the end of the cosine schedule
    weight_decay  : float = 1e-8
    scheduler     : bool  = True
    warmup_epochs : int   = 5
    max_epochs    : int   = 20

@dataclass
class EmbedParams:
    use_neighbors_embeddings          : bool = True
    use_keywords_embeddings           : bool = True
    use_abstract_embeddings           : bool = True
    use_handcrafted_embeddings        : bool = True 
    
    use_jaccard_coefficient           : bool = False
    use_clustering                    : bool = False
    use_adamic_adar_index             : bool = False
    use_preferential_attachment       : bool = False
    use_cn_soundarajan_hopcroft       : bool = False
    use_ra_index_soundarajan_hopcroft : bool = False
    use_sorenson_index                : bool = False   
    # Too big to work
    use_eigenvector_centrality        : bool = False
    use_authors_embeddings            : bool = False 
    use_shortest_path                 : bool = False
    use_common_neighbor_centrality    : bool = False
    

@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """
    dataset_name            : Optional[str]           = "SentenceEmbeddingsFeatures"     # dataset, use <Dataset>Eval for FT
    num_workers             : int                     = 8         # number of workers for dataloadersint
    batch_size              : int                     = 32     # batch_size
    split_val               : float                   = 0.2
    root_dataset            : Optional[str]           = osp.join(os.getcwd(), "input")
    vocab_size              : int                     = 138499

    abstract_embeddings_artifact     : str = 'altegrad-gnn-link-prediction/altegrad_challenge/embeddings.npy:v0'
    keywords_embeddings_artifact     : str = 'altegrad-gnn-link-prediction/altegrad_challenge/keywords-emb-10-sentence-transformers-allenai-specter.npy:v0'
    keywords_artifact                : str = 'altegrad-gnn-link-prediction/altegrad_challenge/keywords-10-sentence-transformers-allenai-specter.npy:v0'
    name_transformer                 : str = 'sentence-transformers/allenai-specter'
    only_create_abstract_embeddings  : bool = False
    only_create_keywords             : bool = False
    nb_keywords                      : int = 10    
    embed_param   : EmbedParams     = EmbedParams()           


@dataclass
class Parameters:
    """base options."""
    hparams       : Hparams         = Hparams()
    data_param    : DatasetParams   = DatasetParams()
    network_param : NetworkParams   = NetworkParams()
    optim_param   : OptimizerParams = OptimizerParams()
    
    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)
            
        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
