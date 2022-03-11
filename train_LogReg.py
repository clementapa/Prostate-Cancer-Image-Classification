import os, zipfile
import wandb
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config.hparams import Parameters
from utils.agent_utils import parse_params


def main():
    parameters = Parameters.parse()

    # initialize wandb instance
    wdb_config = parse_params(parameters)

    wandb_run = wandb.init(
        # vars(parameters),  # FIXME use the full parameters
        config=wdb_config,
        project=parameters.hparams.wandb_project,
        entity=parameters.hparams.wandb_entity,
        allow_val_change=True,
        job_type="train",
        tags=[
            parameters.network_param.network_name,
            parameters.data_param.dataset_name,
            f"patch_size:{parameters.data_param.patch_size}",
            parameters.optim_param.optimizer,
            parameters.data_param.data_provider,
            f"nb_sample:{parameters.data_param.nb_samples}",
            parameters.network_param.classifier_name
        ],
    )
    
    artifact_test = wandb_run.use_artifact('attributes_classification_celeba/dlmi/test_256_1_0.5_score:v0', type='dataset')
    artifact_test_dir = artifact_test.download()
    file_name = os.path.basename(artifact_test_dir.split(':')[0]).replace('_score', '')

    path_to_zip_file = os.path.join(artifact_test_dir, file_name + '.zip')
    if not os.path.exists(os.path.join(artifact_test_dir, file_name)):
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(artifact_test_dir, file_name))
    path_arrays_test = os.path.join(artifact_test_dir, file_name)

    artifact_train = wandb_run.use_artifact('attributes_classification_celeba/dlmi/train_256_1_0.5_score:v0', type='dataset')
    artifact_train_dir = artifact_train.download()
    file_name = os.path.basename(artifact_train_dir.split(':')[0]).replace('_score', '')


    path_to_zip_file = os.path.join(artifact_train_dir, file_name + '.zip')
    if not os.path.exists(os.path.join(artifact_test_dir, file_name)):       
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(artifact_train_dir, file_name))
    path_arrays_train = os.path.join(artifact_train_dir, file_name)

    df_train = pd.read_csv(os.path.join(parameters.data_param.root_dataset, "train.csv"))
    df_test = pd.read_csv(os.path.join(parameters.data_param.root_dataset, "test.csv"))

    name_train_scores = df_train['image_id']+'_score.npy'
    name_test_scores = df_test['image_id']+'_score.npy'
    X_train = np.array([np.mean(np.load(open(os.path.join(path_arrays_train, file), 'rb')), axis=0) for file in name_train_scores])
    X_test = np.array([np.mean(np.load(open(os.path.join(path_arrays_test, file), 'rb')), axis=0) for file in name_test_scores])
    y_train = [df_train[df_train["image_id"] == name.replace('_score.npy', '')]['isup_grade'].values[0] for name in name_train_scores]
    y_train = np.array(y_train)

    clf = LogisticRegression().fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    auc_train = roc_auc_score(y_train, y_train_pred, multi_class="ovr")
    y_test_pred = clf.predict(X_test)


if __name__ == "__main__":
    main()
