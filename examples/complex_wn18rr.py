import sys

from torch import cuda
from torch.optim import Adam
from torchkge import LinkPredictionEvaluator
from torchkge.models import ComplExModel
from torchkge.sampling import TuckerNegativeSampling
from torchkge.utils import BinaryCrossEntropyLoss, DataLoader, load_wn18rr
from tqdm.autonotebook import tqdm
import wandb


def main():
    # Load dataset
    kg_train, _, kg_test = load_wn18rr()

    emb_dim = 1000
    lr = 0.0005
    epochs = 1000
    b_size = 2000
    weight_decay = 1e-5
    n_neg = 25
    cache_dim = 50

    # Define the model
    model = ComplExModel(emb_dim, kg_train.n_ent, kg_train.n_rel)

    # Define Loss
    criterion = BinaryCrossEntropyLoss()

    # Move everything to CUDA if available
    use_cuda = None
    if cuda.is_available():
        cuda.set_device(0)
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()
        use_cuda = 'all'

    # Negative Sampling
    sampler = TuckerNegativeSampling(kg=kg_train, kg_model=model, kg_val=None, kg_test=kg_test, cache_dim=cache_dim,
                                     n_itter=200, k=2, growth=0.1, n_factors=50, n_neg=n_neg)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda=use_cuda)
    evaluator = LinkPredictionEvaluator(model, kg_test)
    iterator = tqdm(range(epochs), unit='epoch')

    # wights and bias configuration
    wandb.login(key="[KEY HERE]")
    wandb.init(project="[PROJECT_NAME_HERE]", entity="[ENTITY_NAME]")
    wandb.config = {
        "scoring_func": "ComplEx",
        "data_set": "wn18rr",
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": b_size,
        "emb_dim": emb_dim
    }

    for epoch in iterator:
        running_loss = 0.0
        # dis_data = []
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]

            n_h, n_t = sampler.corrupt_batch(h, t, r)
            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 25 == 0 and epoch != 0:
            model.normalize_parameters()
            evaluator.evaluate(5)
            evaluator.print_results(k=1, n_digits=4)
            evaluator.print_results(k=3, n_digits=4)
            evaluator.print_results(k=10, n_digits=4)

            hit_1 = round(evaluator.hit_at_k(k=1)[0], 4)
            filt_hit_1 = round(evaluator.hit_at_k(k=1)[1], 4)
            hit_3 = round(evaluator.hit_at_k(k=3)[0], 4)
            filt_hit_3 = round(evaluator.hit_at_k(k=3)[1], 4)
            hit_10 = round(evaluator.hit_at_k(k=10)[0], 4)
            filt_hit_10 = round(evaluator.hit_at_k(k=10)[1], 4)
            mr = int(evaluator.mean_rank()[0])
            filt_mr = int(evaluator.mean_rank()[1])
            mrr = round(evaluator.mrr()[0], 4)
            filt_mrr = round(evaluator.mrr()[1], 4)

            wandb.log(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "hit_1": hit_1,
                    "filt_hit_1": filt_hit_1,
                    "hit_3": hit_3,
                    "filt_hit_3": filt_hit_3,
                    "hit_10": hit_10,
                    "filt_hit_10": filt_hit_10,
                    "mr": mr,
                    "filt_mr": filt_mr,
                    "mrr": mrr,
                    "filt_mrr": filt_mrr
                }
            )
        else:
            wandb.log({"epoch": epoch, "loss": loss})

        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))


if __name__ == "__main__":
    main()

