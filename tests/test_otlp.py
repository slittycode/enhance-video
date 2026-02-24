import os
os.environ["ENABLE_OTLP_TRACING"] = "1"
import upscale_video
upscale_video.init_tracing()

# Add a fake span
with upscale_video.tracer.start_as_current_span("test"):
    pass

print("Exiting...")
