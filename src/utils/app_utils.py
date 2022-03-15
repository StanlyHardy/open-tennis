import yaml
from easydict import EasyDict as edict


class AppUtils(object):

    @classmethod
    def load_config(cls, config_file: str):
        with open(config_file, 'r') as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
            config_data = edict(loaded_config)
            f.close()
        return config_data

    @classmethod
    def load_players(cls, player_file_path: str):
        players_file_stream = open(player_file_path, 'r')
        playersLines = players_file_stream.read().splitlines()
        return playersLines
