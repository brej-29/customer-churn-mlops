from ui.utils import CustomerFeatures, build_payload, classify_churn_risk


def test_build_payload_accepts_integer_like_floats():
    features = CustomerFeatures(
        tenure=12.0,
        monthly_charges=70.0,
        contract_type=0.0,
        has_internet=1.0,
        support_calls=2.0,
        is_senior=0.0,
    )
    payload = build_payload(features)

    assert payload["tenure"] == 12.0
    assert payload["support_calls"] == 2.0
    assert payload["contract_type"] == 0.0
    assert isinstance(payload["monthly_charges"], float)


def test_build_payload_rejects_non_integer_float_for_int_feature():
    features = CustomerFeatures(
        tenure=12.5,
        monthly_charges=70.0,
        contract_type=0.0,
        has_internet=1.0,
        support_calls=2.0,
        is_senior=0.0,
    )
    try:
        build_payload(features)
    except ValueError as exc:
        assert "tenure must be an integer value" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-integer float feature")


def test_classify_churn_risk_buckets():
    low_bucket, _ = classify_churn_risk(0.1)
    medium_bucket, _ = classify_churn_risk(0.5)
    high_bucket, _ = classify_churn_risk(0.9)

    assert low_bucket == "Low"
    assert medium_bucket == "Medium"
    assert high_bucket == "High"