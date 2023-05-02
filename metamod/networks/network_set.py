from metamod.networks import BaseNetwork, LinearNet


class NetworkSet(BaseNetwork):

    def __init__(self, network_class: LinearNet, network_params: dict, n_copies: int = 1):
        self.network_class = network_class
        self.network_params = network_params
        self.n_copies = n_copies
        self.networks = []
        for i in range(n_copies):
            if i > 0:
                W1_0 = self.networks[-1].W1_0.detach().cpu().numpy()
                W2_0 = self.networks[-1].W2_0.detach().cpu().numpy()
                self.network_params['W1_0'] = W1_0
                self.network_params['W2_0'] = W2_0
            self.networks.append(network_class(**network_params))
