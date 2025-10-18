"""
Unit tests for the labeler functionality in scales.
"""

import pytest
import numpy as np
from dapple.scales import default_labeler, ScaleDiscreteLength, xdiscrete


class TestDefaultLabeler:
    """Tests for the default_labeler function."""

    def test_empty_sequence(self):
        """Empty sequences should return empty list."""
        result = default_labeler([])
        assert result == []

    def test_integers(self):
        """Integers should be formatted without decimal points."""
        result = default_labeler([1, 2, 3, 4, 5])
        assert result == ["1", "2", "3", "4", "5"]

    def test_floats_with_matching_precision(self):
        """Floats should be formatted with matching precision."""
        result = default_labeler([1.1, 2.2, 3.3, 4.4, 5.5])
        assert all(label.count(".") == 1 for label in result)
        assert all(len(label.split(".")[1]) == 1 for label in result)

    def test_mixed_precision_numbers(self):
        """Numbers with varying precision should use consistent precision."""
        result = default_labeler([1.0, 1.5, 1.23, 1.456, 1.789])
        # All should have same number of decimal places
        decimals = [len(label.split(".")[1]) for label in result]
        assert len(set(decimals)) == 1
        # Should use 3 decimal places (max needed)
        assert decimals[0] == 3

    def test_small_decimals(self):
        """Small decimal numbers should be formatted appropriately."""
        result = default_labeler([0.1, 0.2, 0.3, 0.4, 0.5])
        assert result == ["0.1", "0.2", "0.3", "0.4", "0.5"]

    def test_strings(self):
        """String values should be passed through with str()."""
        result = default_labeler(["apple", "banana", "cherry"])
        assert result == ["apple", "banana", "cherry"]

    def test_mixed_types(self):
        """Mixed types should fall back to str() conversion."""
        result = default_labeler([1, "two", 3.0, None])
        assert result == ["1", "two", "3.0", "None"]

    def test_numpy_types(self):
        """Numpy numeric types should be formatted with precision."""
        result = default_labeler([np.float64(1.1), np.float64(2.2), np.float64(3.3)])
        assert all(label.count(".") == 1 for label in result)

    def test_complex_floats(self):
        """Very small or very large floats should be handled."""
        result = default_labeler([0.00001, 0.00002, 0.00003])
        # Should all have consistent formatting
        assert len(result) == 3
        assert all(isinstance(label, str) for label in result)


class TestScaleDiscreteLabelingIntegration:
    """Integration tests for labeler in ScaleDiscreteLength."""

    def test_discrete_scale_with_numeric_values(self):
        """Discrete scale should use labeler for numeric values."""
        scale = xdiscrete(values=[1.1, 2.2, 3.3])
        scale.finalize()
        labels, _ = scale.ticks()
        # All labels should have matching precision
        assert all("." in str(label) for label in labels)

    def test_discrete_scale_with_string_values(self):
        """Discrete scale should use labeler for string values."""
        scale = xdiscrete(values=["a", "b", "c"])
        scale.finalize()
        labels, _ = scale.ticks()
        assert list(labels) == ["a", "b", "c"]

    def test_discrete_scale_with_custom_labeler(self):
        """Discrete scale should accept custom labeler function."""

        def custom_labeler(values):
            return [f"Item {v}" for v in values]

        scale = xdiscrete(values=[1, 2, 3], labeler=custom_labeler)
        scale.finalize()
        labels, _ = scale.ticks()
        assert list(labels) == ["Item 1", "Item 2", "Item 3"]

    def test_discrete_scale_labeler_called_once(self):
        """Labeler should be called with all values at once, not one by one."""
        call_count = 0
        received_values = []

        def counting_labeler(values):
            nonlocal call_count, received_values
            call_count += 1
            received_values.extend(values)
            return [str(v) for v in values]

        scale = xdiscrete(values=[1, 2, 3, 4, 5], labeler=counting_labeler)
        # Labeler should be called once during initialization
        assert call_count == 1
        assert received_values == [1, 2, 3, 4, 5]

    def test_discrete_scale_fit_values_batched(self):
        """Labeler should be called once for all new fitted values."""
        call_count = 0
        all_calls = []

        def tracking_labeler(values):
            nonlocal call_count
            call_count += 1
            all_calls.append(list(values))
            return [f"L{v}" for v in values]

        from dapple.scales import UnscaledValues

        scale = xdiscrete(values=[1, 2], labeler=tracking_labeler)
        assert call_count == 1  # Initial call

        # Fit new values
        scale.fit_values(UnscaledValues("x", [3, 4, 5]))
        assert call_count == 2  # One more call for new values
        assert all_calls[1] == [3, 4, 5]

    def test_discrete_scale_mapping_values(self):
        """Discrete scale with mapping should label keys, not targets."""
        scale = xdiscrete(values={"a": 1, "b": 2, "c": ("Custom", 3)})
        scale.finalize()
        labels, _ = scale.ticks()
        # "a" and "b" should be labeled with their keys as strings: "a", "b"
        # "c" should use custom label "Custom"
        label_set = set(str(label) for label in labels)
        assert "a" in label_set
        assert "b" in label_set
        assert "Custom" in label_set


class TestNumberLabelingPrecision:
    """Tests for numeric precision in labels."""

    def test_trailing_zeros_removed(self):
        """Unnecessary trailing zeros should be removed."""
        result = default_labeler([1.0, 2.0, 3.0])
        assert result == ["1", "2", "3"]

    def test_minimal_precision_preserved(self):
        """Minimum necessary precision should be preserved."""
        result = default_labeler([1.0, 1.1, 1.2])
        assert result == ["1.0", "1.1", "1.2"]

    def test_max_precision_within_set(self):
        """Maximum precision in set determines precision for all."""
        result = default_labeler([1.0, 1.5, 1.123])
        # All should have 3 decimal places to match 1.123
        assert all(len(label.split(".")[1]) == 3 for label in result)
