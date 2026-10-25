
import os

from mohh.core.params import Params
from mohh.core.utilities.paths import project_root

def instance_paths(train: bool):
    """
    Access the datasets folder and load the files inside.
    """

    params = Params()

    dataset_path = params["DATASET_PATH"]
    if dataset_path:
        dataset_path = os.path.abspath(os.path.expanduser(str(dataset_path)))
    else:
        dataset_path = os.path.join(project_root(), "datasets",
                                    params["DATASET"])

    if train:
        target_path = os.path.join(dataset_path, "train")
    else:
        target_path = os.path.join(dataset_path, "test")
    
    if not os.path.isdir(target_path):
        raise FileNotFoundError(
            f"no existe la carpeta de instancias: {target_path}")

    instances = sorted(
        os.path.join(target_path, name)
        for name in os.listdir(target_path)
        if name.endswith(".txt"))

    return instances
