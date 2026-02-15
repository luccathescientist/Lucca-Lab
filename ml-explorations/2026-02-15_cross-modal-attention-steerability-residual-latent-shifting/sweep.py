import torch
import matplotlib.pyplot as plt


def run_once(seq_len=1024, num_heads=32, seed=0, steering_lambda=0.0):
    g = torch.Generator().manual_seed(seed)
    saliency = torch.abs(torch.randn(seq_len, generator=g))
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-12)

    original_attn = torch.softmax(torch.randn(num_heads, seq_len, seq_len, generator=g) * 0.1, dim=-1)

    bias = saliency.unsqueeze(0).unsqueeze(0).expand(num_heads, seq_len, seq_len)
    steered_attn = torch.softmax(torch.log(original_attn + 1e-9) + steering_lambda * bias, dim=-1)

    entropy_original = -torch.sum(original_attn * torch.log(original_attn + 1e-9), dim=-1).mean()
    entropy_steered = -torch.sum(steered_attn * torch.log(steered_attn + 1e-9), dim=-1).mean()
    kl_div = torch.sum(original_attn * (torch.log(original_attn + 1e-9) - torch.log(steered_attn + 1e-9)), dim=-1).mean()

    ent_red = ((entropy_original - entropy_steered) / entropy_original).item() * 100
    return ent_red, kl_div.item()


def main():
    lambdas = [0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
    ent_reds = []
    kls = []

    # average over a few seeds for stability
    seeds = [0, 1, 2]

    for lam in lambdas:
        ents = []
        kls_i = []
        for s in seeds:
            er, kl = run_once(seed=s, steering_lambda=lam)
            ents.append(er)
            kls_i.append(kl)
        ent_reds.append(sum(ents)/len(ents))
        kls.append(sum(kls_i)/len(kls_i))

    fig, ax1 = plt.subplots(figsize=(9,5.5))
    ax1.plot(lambdas, ent_reds, marker='o', label='Entropy reduction (%)')
    ax1.set_xlabel('steering_lambda (residual bias strength)')
    ax1.set_ylabel('Attention entropy reduction (%)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(lambdas, kls, marker='s', color='orange', label='KL divergence')
    ax2.set_ylabel('KL(original || steered)')

    # combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.title('Tradeoff: Steerability vs Distribution Shift')
    plt.tight_layout()
    plt.savefig('lambda_tradeoff.png', dpi=160)

    # Print a small table
    for lam, er, kl in zip(lambdas, ent_reds, kls):
        print(f"lambda={lam:>4}: entropy_reduction={er:>7.4f}%  KL={kl:.6f}")


if __name__ == '__main__':
    main()
