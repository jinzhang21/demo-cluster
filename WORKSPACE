workspace(name = "mlflow_endpoint_service")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Python dependencies for both services
load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
    name = "requirements",
    requirements = "//:requirements.txt",  # Load the Python dependencies here
)

# Bazel rules for Kubernetes (if needed for direct Kubernetes integration)
# These are optional if you're automating deployments with scripts.
http_archive(
    name = "k8s",
    urls = ["https://github.com/bazelbuild/rules_k8s/archive/refs/tags/0.5.0.tar.gz"],
    strip_prefix = "rules_k8s-0.5.0",
)

