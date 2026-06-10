"""retrieve connector-family tests."""

import pytest

# The contract suite lives in a non-test module imported by the concrete test
# files; register it so its inherited asserts get pytest's assertion rewriting.
pytest.register_assert_rewrite("tests.connectors.retrieve.retrieve_contract")
