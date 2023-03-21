import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from trialexp.process.pycontrol.session_analysis import add_trial_nb

def test_add_trial_nb():
    # Create test data
    df = pd.DataFrame({'time': [0.1, 0.5, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5]})
    trigger_time = [1.0, 2.0, 3.0]
    trial_window = (-0.5, 0.5)

    # Call function under test
    df_trial, valid_trigger_time = add_trial_nb(df, trigger_time, trial_window)

    # Check that trial numbers were added correctly
    expected_trial_numbers = [np.nan, 1, 1, 2, 2, 3, 3, np.nan]
    assert_array_equal(df_trial['trial_nb'].values, expected_trial_numbers)

    # Check that valid trigger times were returned correctly
    expected_valid_trigger_time = np.array([1.0, 2.0, 3.0])
    assert_array_equal(valid_trigger_time, expected_valid_trigger_time)

    # Check that overlapping trials are not allowed
    df = pd.DataFrame({'time': [0.1, 0.5, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5]})
    trigger_time = [1.0, 2.0, 2.5, 3.5]
    trial_window = (-0.5, 0.5)
    df_trial, valid_trigger_time = add_trial_nb(df, trigger_time, trial_window)
    expected_trial_numbers = [np.nan, 1, 1, 2, 2, np.nan, 3, 3]
    assert_array_equal(df_trial['trial_nb'].values, expected_trial_numbers)
