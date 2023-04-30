from metamod.networks import BaseNetwork, LinearNet


class NetworkSet(BaseNetwork):

    def __init__(self, network_class: LinearNet, network_params: dict, n_copies: int = 1):
        self.network_class = network_class
        self.network_params = network_params
        self.n_copies = n_copies
        self.networks = [network_class(**network_params) for _ in range(n_copies)]
