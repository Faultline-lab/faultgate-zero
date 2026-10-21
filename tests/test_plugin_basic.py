def test_plugin_loads_with_no_model_calls(pytester):
    pytester.makepyfile("def test_nothing():\n    assert True\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["faultgate-zero: 0 runs checked"])


def test_plugin_loads_in_a_subprocess_through_the_entry_point(pytester):
    pytester.makepyfile("def test_nothing():\n    assert True\n")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["faultgate-zero: 0 runs checked"])
