import yaml
from easydict import EasyDict as edict


class AppUtils(object):

    @classmethod
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
            config_data = edict(loaded_config)
            f.close()
        return config_data
