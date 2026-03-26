from enhance_video import telemetry


def test_init_tracing_updates_package_tracer_reference():
    telemetry.init_tracing()

    assert telemetry.tracer is not None
