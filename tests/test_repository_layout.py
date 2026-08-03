from tools.audit_repository_layout import audit


def test_every_subfolder_has_at_most_100_recursive_files():
    _, violations = audit()
    assert not violations, violations
