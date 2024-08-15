import torch


def gaussian_noise(x, mean=0, std=0.1):
    noise = torch.normal(mean, std, x.size()).to(x.device)
    noisy_x = x + noise
    return noisy_x


def markov_noise(x, transition_matrix):
    num_bins, num_frames = x.shape
    noisy_x = x.clone()
    state = torch.randint(0, num_bins, (1,)).item()

    for t in range(num_frames):
        noisy_x[:, t] = x[:, t] * (1 + transition_matrix[state])
        state = torch.multinomial(transition_matrix[state], 1).item()

    return noisy_x


def random_walk_noise(x, step_size=0.1):
    noisy_x = x.clone()
    random_walk = torch.FloatTensor(x.size()).uniform_(-step_size, step_size).to(x.device)
    noisy_x += random_walk
    return noisy_x


def spectral_folding(x, fold_frequency=0.5):
    num_bins, num_frames = x.shape
    fold_bin = int(fold_frequency * num_bins)

    folded_x = x.clone()
    folded_x[:fold_bin, :] += x[-fold_bin:, :]
    folded_x[-fold_bin:, :] = 0

    return folded_x
