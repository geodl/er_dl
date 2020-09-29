from pathlib import Path
import pickle


class CommonConfig:
    root_dir = Path(__file__).resolve().parents[1]
    app_dir = root_dir / 'project' / 'app'


def get_cmap():
    with open(CommonConfig.root_dir / 'project' / 'utils' / 'cmap.pkl', 'rb') as fin:
        cmap_pkl = pickle.load(fin)
    return cmap_pkl


common_config = CommonConfig()
cmap = get_cmap()
