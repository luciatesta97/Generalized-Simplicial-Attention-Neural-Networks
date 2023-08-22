class CollaborationComplex(torch.utils.data.Dataset):
    def __init__(self, pct_miss, order, num_exp, eps, kappa,
                 device,
                 starting_node=150250,
                 data_path=r"data/collaboration_complex",):

        assert order >= 0
        assert pct_miss in range(10, 60, 10)
        self.incidences = np.load('{}/{}_boundaries.npy'.format(data_path,
                                                                starting_node),
                                  allow_pickle=True)

        # workaround order == len(self.incidences)
        # is not taken into account at the moment since is higher than
        # the maximum number used in the experiments.
        """
        Lup = torch.from_numpy(
            (self.incidences[order] @ self.incidences[order].T).A)
        if order == 0:
            Ldo = torch.zeros_like(Lup)
        else:
            Ldo = torch.from_numpy(
                (self.incidences[order-1].T @ self.incidences[order-1]).A)
        """
        Lup = np.load("{}/{}_laplacians_up.npy".format(data_path, starting_node), allow_pickle=True)[order]
        Ldo = np.load("{}/{}_laplacians_down.npy".format(data_path, starting_node), allow_pickle=True)[order]
        L = np.load("{}/{}_laplacians.npy".format(data_path, starting_node), allow_pickle=True)[order]

        Ldo = coo2tensor(normalize2(L, Ldo ,half_interval=True)).to_dense()
        Lup = coo2tensor(normalize2(L, Lup ,half_interval=True)).to_dense()
        L1 = coo2tensor(normalize2(L, L ,half_interval=True)).to_dense()

        self.L = (Ldo, Lup, compute_projection_matrix(
            L1, eps=eps, kappa=kappa))

        observed_signal = np.load('{}/{}_percentage_{}_input_damaged_{}.npy'.format(
            data_path, starting_node, pct_miss, num_exp), allow_pickle=True)
        observed_signal = [torch.tensor(
            list(signal.values()), dtype=torch.float) for signal in observed_signal]

        #
        target_signal = np.load(
            '{}/{}_cochains.npy'.format(data_path, starting_node), allow_pickle=True)
        target_signal = [torch.tensor(
            list(signal.values()), dtype=torch.float) for signal in target_signal]

        masks = np.load('{}/{}_percentage_{}_known_values_{}.npy'.format(data_path, starting_node, pct_miss,
                        num_exp), allow_pickle=True)  # positive mask= indices that we keep ##1 mask #entries 0 degree
        masks = [torch.tensor(
            list(mask.values()), dtype=torch.long) for mask in masks]

        self.X = observed_signal[order].reshape(-1,1).to(device)
        self.y = target_signal[order].to(device)
        self.n = len(self.X)
        self.mask = masks[order].to(device)

    def __getitem__(self, index):
        return self.X, self.y

    def __len__(self):
        # Returns length
        return 1000