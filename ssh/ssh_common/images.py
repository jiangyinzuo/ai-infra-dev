"""Docker image repositories supported by listing and container creation."""

IMAGE_REPOSITORIES = ("vllm-ascend", "quay.io/ascend/vllm-ascend")


def image_list_command():
    command = ["docker", "images"]
    for repository in IMAGE_REPOSITORIES:
        command += ["--filter", "reference=" + repository + ":*"]
    return command
