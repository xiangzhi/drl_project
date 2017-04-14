#!/usr/bin/env python


from test.test_preprocessors import PreprocessorsTestMethods

import rosunit
rosunit.unitrun('drl_proj','dqn_test',PreprocessorsTestMethods)