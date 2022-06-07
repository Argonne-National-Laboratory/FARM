
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED


import sys
import importlib
import platform
import xml.etree.ElementTree as ET
import os
from os import path

# Get RAVEN location
config = path.abspath(path.join(path.dirname(__file__),'..','..','.ravenconfig.xml'))
# print(" ** config_LOC=",config)
if not path.isfile(config):
  raise IOError(
      f'FARM config file not found at "{config}"! Has FARM been installed as a plugin in a RAVEN installation?'
  )
loc = ET.parse(config).getroot().find('FrameworkLocation')
assert loc is not None and loc.text is not None
RAVEN_LOC = path.abspath(path.dirname(loc.text)) # /raven

# get RAVEN base testers
TESTER_LOC = os.path.join(RAVEN_LOC, 'scripts', 'TestHarness', 'testers') # /testers
sys.path.append(TESTER_LOC)
from RavenFramework import RavenFramework as RavenTester
sys.path.pop()



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
    # NOTE: self.driver is RAVEN driver (e.g. /path/to/Driver.py)

  def check_runnable(self): # modified from raven\scripts\TestHarness\testers\RavenFramework.py, 
    """
      Checks if this test can run.
      @ In, None
      @ Out, check_runnable, boolean, if True can run this test.
    """

    ## OS
    if len(self.specs['skip_if_OS']) > 0:
      skipOs = [x.strip().lower() for x in self.specs['skip_if_OS'].split(',')]
      # get simple-name platform (options are Linux, Windows, Darwin, or SunOS that I've seen)
      currentOs = platform.system().lower()
      # replace Darwin with more expected "mac"
      if currentOs == 'darwin':
        currentOs = 'mac'
      if currentOs in skipOs:
        self.set_skip('skipped (OS is "{}")'.format(currentOs))
        return False
    
    ## HERON
    # Get HERON location
    plugin_handler_dir = path.join(RAVEN_LOC, 'scripts')
    sys.path.append(plugin_handler_dir)
    plugin_handler = importlib.import_module('plugin_handler')
    sys.path.pop()
    HERON_LOC = plugin_handler.getPluginLocation('HERON') # /HERON
    if HERON_LOC is None:
      self.set_skip('skipped (HERON is not installed)')
      return False
    else:
      self.heron_driver = os.path.join(HERON_LOC, 'heron')
    return True

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

    # print(" ** cmd = ", cmd)

    return cmd

