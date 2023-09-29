from typing import Dict, List, Literal, Optional, Union, Tuple
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
        assert trainer_config['output_dir'] is not None
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

    def run(self, printing: Literal["devnull", "pipe", "print"] = "print") -> Tuple[subprocess.CompletedProcess, List[str]]:
        """
        Runs the training script with the given arguments.

        Args:
        printing (Literal["devnull", "pipe", "print"], optional): The printing behavior of the
        training script. If "devnull", then the output is redirected to /dev/null. If "pipe",
        then the output is piped to the parent process. If "print", then the output is printed
        to stdout. Defaults to "print".

        Returns:
        Tuple[subprocess.CompletedProcess, List[str]]: A tuple of the subprocess.CompletedProcess
        object and a list of paths to checkpoints that were saved during training.
        """
        bash_cmd = self.to_bash()

        if printing == "print":
            print(bash_cmd)
            stdout = None
            stderr = None
        elif printing == "devnull":
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        elif printing == "pipe":
            stdout = subprocess.PIPE
            stderr = subprocess.STDOUT

        p = subprocess.run(
            bash_cmd,
            shell=True,
            stdout=stdout,
            stderr=stderr,
            cwd=PROJ_DIR,
            encoding='utf-8'
        )

        if p.returncode != 0:
            return p, []

        # check output_dir for checkpoint directories
        checkpoints = []
        output_dir = self.trainer_config['output_dir']
        assert isinstance(output_dir, str)
        for filename in os.listdir(output_dir):
            if filename.startswith('checkpoint'):
                checkpoints.append(os.path.join(output_dir, filename))

        return p, checkpoints


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
            'epochs': 1,
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
    res, dirs = config.run(printing="print")
    print(res.returncode)
    print(dirs)
