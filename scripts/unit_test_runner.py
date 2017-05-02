#!/usr/bin/env python

import unittest
from test.test_preprocessors import PreprocessorsTestMethods
from test.test_memory import ActionMemoryTestMethods

# import rosunit
# rosunit.unitrun('drl_proj','dqn_test',PreprocessorsTestMethods)
#PreprocessorsTestMethods.main()

suite = unittest.TestLoader().loadTestsFromTestCase(PreprocessorsTestMethods)
unittest.TextTestRunner(verbosity=2).run(suite)
suite = unittest.TestLoader().loadTestsFromTestCase(ActionMemoryTestMethods)
unittest.TextTestRunner(verbosity=2).run(suite)