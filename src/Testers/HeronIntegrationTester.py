
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED


import sys
import importlib
import platform
import xml.etree.ElementTree as ET
import os
from os import path

# # get heron utilities
# HERON_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(" ** HERON_LOC=",HERON_LOC)
# sys.path.append(HERON_LOC)
# import _utils as hutils
# sys.path.pop()

# # get RAVEN base testers
# RAVEN_LOC = hutils.get_raven_loc()
# TESTER_LOC = os.path.join(RAVEN_LOC, '..', 'scripts', 'TestHarness', 'testers')
# sys.path.append(TESTER_LOC)
# from RavenFramework import RavenFramework as RavenTester
# sys.path.pop()

# Get RAVEN location
config = path.abspath(path.join(path.dirname(__file__),'..','..','.ravenconfig.xml'))
# print(" ** config_LOC=",config)
if not path.isfile(config):
  raise IOError(
      f'FARM config file not found at "{config}"! Has FARM been installed as a plugin in a RAVEN installation?'
  )
loc = ET.parse(config).getroot().find('FrameworkLocation')
assert loc is not None and loc.text is not None
RAVEN_LOC = path.abspath(path.dirname(loc.text)) #/raven
print(" ** RAVEN_LOC=",RAVEN_LOC)

# get RAVEN base testers
TESTER_LOC = os.path.join(RAVEN_LOC, 'scripts', 'TestHarness', 'testers')
print(" ** TESTER_LOC=",TESTER_LOC)
sys.path.append(TESTER_LOC)
from RavenFramework import RavenFramework as RavenTester
sys.path.pop()


# Get HERON location
plugin_handler_dir = path.join(RAVEN_LOC, 'scripts')
sys.path.append(plugin_handler_dir)
plugin_handler = importlib.import_module('plugin_handler')
sys.path.pop()
HERON_LOC = plugin_handler.getPluginLocation('HERON')



class HeronIntegration(RavenTester):
  """
    Defines testing mechanics for HERON integration tests; that is, tests that
    run the full HERON-RAVEN suite on a case.
  """

  # TODO extend get_valid_params?

  def __init__(self, name, param):
    """
      Constructor.
      @ In, name, str, name of test
      @ In, params, dict, test parameters
      @ Out, None
    """
    RavenTester.__init__(self, name, param)
    # self.heron_driver = os.path.join(HERON_LOC, '..', 'heron')
    self.heron_driver = os.path.join(HERON_LOC, 'heron')
    # NOTE: self.driver is RAVEN driver (e.g. /path/to/Driver.py)

  def get_command(self):
    """
      Gets the command line commands to run this test
      @ In, None
      @ Out, cmd, str, command to run
    """
    cmd = ''
    python = self._get_python_command()
    test_loc = os.path.abspath(self.specs['test_dir'])
    # HERON expects to be run in the dir of the input file currently, TODO fix this
    cmd += ' cd {loc} && '.format(loc=test_loc)
    # clear the subdirectory if it's present
    cmd += ' rm -rf *_o/ && '
    # run HERON first
    heron_inp = os.path.join(test_loc, self.specs['input'])
    # Windows is a little different with bash scripts
    if platform.system() == 'Windows':
      cmd += ' bash.exe '
    cmd += f' {self.heron_driver} {heron_inp} && '
    # then run "outer.xml"
    raven_inp = os.path.abspath(os.path.join(os.path.dirname(heron_inp), 'outer.xml'))
    cmd += f' {python} {self.driver} {raven_inp}'

    print(" ** cmd = ", cmd)

    return cmd

