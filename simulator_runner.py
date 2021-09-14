import os
from snntoolbox.utils.utils import import_configparser
from snntoolbox.simulation.target_simulators.brian2_target_sim import SNN

filepath_config = "./ann-to-snn.ini"

configparser = import_configparser()

config = configparser.ConfigParser()
config.optionxform = str
config.read(os.path.abspath(os.path.join(filepath_config)))

snn = SNN(config)