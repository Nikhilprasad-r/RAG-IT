class RagkitError(Exception):
    """Base class for every error ragkit raises on purpose."""


class ConfigError(RagkitError):
    """A project config or agent manifest is missing, malformed or inconsistent."""


class ProjectNotFoundError(ConfigError):
    """No ragkit.yaml was found in the directory or any of its parents."""


class PluginNotFoundError(RagkitError):
    """A plugin was requested by a name nothing registered."""
