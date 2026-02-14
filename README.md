# GrasAD
This is the experimental code for the paper "GrasAD: Gradient Similarity and Accumulated Distribution-Based Device Selection for Asynchronous Federated Learning".

## Environment Installation

### 1.  Install FederatedScope
Visit **alibaba/FederatedScope** (https://github.com/alibaba/FederatedScope) and check the **Installation** section in the README.md.
### 2.  Clone this project code
Place the **GrasAD** code into the corresponding extension directory of FederatedScope.
### 3.  Download and prepare the dataset
(1) Download the LEAF dataset. Run python federatedscope/main.py --cfg /scripts/example_configs/femnist.yaml, setting the data.type in the yaml file to femnist, celeba, shakespeare, and synthetic respectively. In this way, we obtain the corresponding experimental datasets in the local federatedscope/data directory. For details, refer to the dataset description in /federatedscope/core/configs.

**Note that the LEAF dataset provided by the official source is already pre-processed into a Non-IID dataset by user. Therefore, the degree of Non-IID cannot be directly adjusted through parameters in FederatedScope.**

(2) Modify the Non-IID degree of the LEAF dataset. Run leaf_celeba.py, leaf_femnist.py, and leaf_shakes.py located in federatedscope/contrib/data respectively; we will obtain centralized datasets with the default user partitioning removed.

## Run the script

Our experiments include two types of datasets: directly using the LEAF dataset (where the Non-IID degree cannot be adjusted) and using the LEAF dataset with user partitioning removed (where the Non-IID degree can be adjusted). 

(1) Taking running GrasAD on the CelebA dataset as an example, the running command is:

python federatedscope/main.py --cfg scripts/noniidagg_algorithms/celeba_usrsele_grasad_vgg11_alpha_0_5_leaf.yaml

The difference in their usage lies in the different data names specified in the YAML configuration file. For the LEAF dataset with default user partitioning, set data.type=celeba. For the LEAF dataset with custom user partitioning, set data.type=leaf_celeba, with data.spliiter='lda'.

(2) We can modify parameters such as data, algorithm, client_num, and total_round_num in any YAML configuration file, or we can run different pre-configured YAML files.

(3) The correspondence between the algorithm names in the code and those in the paper is shown in the table below.

| Code Algorithm Name | Paper Algorithm Name |
| :--- | :--- |
| `usrsele` | GrasAD |
| `usrsele_compa_aouprior` | AouPrio |
| `usrsele_compa_freqandage` | FreAgg |
| `usersele_compa_hfl` | HFL |
| `usrsele_compa_kafl` | KAFL |
| `usrsele_compa_schandagg` | SchAgg |
| `usrsele_compa_wkafl` | WKAFL |

**Note that each time you use an algorithm, you must ensure that "asy.use == True" to confirm asynchronous federated updates.** For example, when using the HFL algorithm, you must set: "asy.use == True", and "usrsele_compa_hfl:.use == True". However, when you want to use AFedAvg, just set asy.use=True.

# Acknowledgement
We would like to thank the authors for releasing the public repository: FederatedScope-LLM.
