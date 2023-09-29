from typing import Any, Dict, List, Optional, Union
import subprocess
import os


PROJ_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainingAPI:
    def __init__(
            self,
            trainer_config: Dict[str, Optional[Union[str, int, float, bool]]],
            gpu_ids: Optional[List[int]] = None,
            master_port=29500,
    ):
        """
        Creates a TrainingAPI object that can be used to train a model.

        Args:
            trainer_config (Dict[str, Optional[Union[str, int, float, bool]]]): A dictionary of
            each of the arguments that can be passed to the trainer. The keys are the names of the
            arguments and the values are the values of the arguments. If the value is None, then
            the argument is passed without a value (e.g. --no_fp16). If the value is not None, then
            the argument is passed with a value (e.g. --learning_rate=2e-5).
            gpu_ids (Optional[List[int]], optional): A list of GPU ids to use for training. If
            None, then all available GPUs are used. Defaults to None.
            master_port (int, optional): The port to use for the master process. Defaults to 29500.
        """
        if gpu_ids is None:
            import torch
            # figures out how many GPUs are available
            gpu_ids = list(range(torch.cuda.device_count()))
        self.gpu_ids = gpu_ids
        self.trainer_config = trainer_config
        self.master_port = master_port

    def to_bash(self) -> str:
        gpu_ids_str = ','.join([str(gpu_id) for gpu_id in self.gpu_ids])
        bash_cmd = f'CUDA_VISIBLE_DEVICES={gpu_ids_str} python3 -m torch.distributed.launch ' \
            + f'--nproc_per_node {len(self.gpu_ids)} ' \
            + f'--master_port {self.master_port} ' \
            + f'{os.path.join(PROJ_DIR, "main.py")} ' \
            + ' '.join([f'--{k}={v}' if v is not None else f'--{k}' for k,
                        v in self.trainer_config.items()])
        return bash_cmd

    def run(self, verbose: bool = True) -> subprocess.CompletedProcess:
        """
        Runs the training script with the given arguments.

        Args:
            verbose (bool, optional): Whether to print stdout and stderr. Defaults to True.

        Returns:
            subprocess.CompletedProcess: The result of the training script.
        """
        bash_cmd = self.to_bash()
        if verbose:
            print(bash_cmd)

        return subprocess.run(
            bash_cmd,
            shell=True,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
            cwd=PROJ_DIR,
            encoding='utf-8'
        )


if __name__ == "__main__":
    # example usage
    config = TrainingAPI(
        gpu_ids=[4, 5, 6],
        trainer_config={
            'model_path': 'bigcode/starcoderbase-1b',
            'dataset_name': 'nuprl/MultiPL-T',
            'split': 'lua',
            'output_dir': '/tmp/testing',
            'seq_length': 1024,
            'epochs': 2,
            'batch_size': 16,
            'gradient_accumulation_steps': 8,
            'learning_rate': 2e-5,
            'num_warmup_steps': 10,
            'num_workers': 4,
            'no_fp16': None,
            'bf16': None,
            'perc_valid_set': 0.05,
            'save_total_limit': 20,
        }
    )
    res = config.run(verbose=True)
