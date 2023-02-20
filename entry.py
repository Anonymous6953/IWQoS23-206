import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models, ModelRegistryBase
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.utils.testing.core_stubs import get_branin_search_space, get_branin_experiment
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import get_logger

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from botorch.test_functions import BraninCurrin
from gpytorch.mlls import ExactMarginalLogLikelihood


from utils.dataset import TZSDataset, TZSTestDataset
from utils.evaluate import best_f1_score_with_point_adjust
from models.Parameters import parameters
from models import Donut_train, Donut_test
from utils.obj_fn import mse_obj, nf_obj


# setup_seed and logger
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
setup_seed(2023)
logger = get_logger(name="AutoKAD")



# Load configuration
with open("./conf.yml", 'r') as f:
    config = yaml.load(f, yaml.FullLoader)
device = config['device']



train_dataset = TZSDataset(config['train_kpi_path'])
test_dataset = TZSTestDataset(config['test_kpi_path'])



def evaluate(x_raw: np.ndarray, x_est: np.ndarray, labels: np.ndarray):
    anomaly_scores = np.abs(x_raw - x_est)
    mse = mse_obj(x_raw, x_est)
    nf = nf_obj(x_est)
    mse_nf = mse + nf

    res = best_f1_score_with_point_adjust(labels, anomaly_scores)

    res = {
        'precision': (res['p'], 0.0),
        'recall': (res['r'], 0.0),
        'f1': (res['r'], 0.0),
        'mse': (mse, 0.0),
        'nf': (nf, 0.0),
        'mse_nf': (mse_nf, 0.0)
    }

    return res


# Test dataset and labels are only used for experiment purpose.
def optimize_loop(params=None):
    win_len = params['win_len']
    batch_size = params.get("batch_size")

    train_dataset.set_win_len(win_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model_name = params.get("model")
    if model_name == "Donut":
        train = Donut_train
    elif model_name == "LSTM":
        train = LSTM_

    try:
        model = Donut_train(
            params=params, 
            dataloader=train_dataloader, 
            device=device
            )

        test_dataset.set_win_len(win_len)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

        x_raw, x_est, loss, labels = Donut_test(
            model=model,
            dataloader=test_dataloader,
            device=device
        )

        res = evaluate(x_raw, x_est, labels)
        res['loss'] = (loss, 0.0)

    except Exception as e:
        print("Someting went wrong!")
        res = {
            'precision': (0.0, 0.0),
            'recall': (0.0, 0.0),
            'f1': (0.0, 0.0),
            'mse': (0x7fffffff, 0.0),
            'nf': (0x7fffffff, 0.0),
            'mse_nf': (0x7fffffff, 0.0),
        }

    logger.info(f"performance: {res}")
    return res

gs = GenerationStrategy(
    steps=[
        # 1. Initialization step (does not require pre-existing data and is well-suited for 
        # initial sampling of the search space)
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
            min_trials_observed=3, # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            model_kwargs={},  # Any kwargs you want passed into the model
            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.GPEI,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        ),
    ]
)


# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name="AutoKAD_experiment",
    parameters=parameters,
    objectives={
        "mse_nf": ObjectiveProperties(minimize=True),
    },
    tracking_metric_names=["mse", "f1", "precision", "recall", "loss", "mse_nf", "nf"]
)
# Setup a function to evaluate the trials


for i in range(10):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=optimize_loop(parameters))

logger.info("Finished!")