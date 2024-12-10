import nodetools.configuration.configuration as config

class BaseUtilities:
    """Base class for shared functionality between utilities"""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.__class__._initialized:
            self.network_config = config.get_network_config()
            self.node_config = config.get_node_config()
            self.primary_endpoint = self._get_primary_endpoint()
            self.__class__._initialized = True

    def _get_primary_endpoint(self):
        """Determine endpoint with fallback logic"""
        return (
            self.network_config.local_rpc_url 
            if config.RuntimeConfig.HAS_LOCAL_NODE and self.network_config.local_rpc_url is not None
            else self.network_config.public_rpc_url
        )