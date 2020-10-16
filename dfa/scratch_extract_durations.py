from pathlib import Path

import torch
import numpy as np
import tqdm

from dfa.audio import Audio
from dfa.dataset import new_dataloader
from dfa.extract_durations import extract_durations_with_dijkstra
from dfa.model import Aligner
from dfa.paths import Paths
from dfa.text import Tokenizer
from dfa.utils import read_metafile


def to_device(batch: dict, device: torch.device) -> tuple:
    tokens, mel, tokens_len, mel_len = batch['tokens'], batch['mel'], \
                                       batch['tokens_len'], batch['mel_len']
    tokens, mel, tokens_len, mel_len = tokens.to(device), mel.to(device), \
                                       tokens_len.to(device), mel_len.to(device)
    return tokens, mel, tokens_len, mel_len


if __name__ == '__main__':

    checkpoint = torch.load('/Users/cschaefe/dfa_checkpoints/latest_model.pt', map_location=torch.device('cpu'))
    data_dir = '/tmp/dfa_data'

    config = checkpoint['config']
    symbols = checkpoint['symbols']
    audio = Audio(**config['audio'])

    # override paths
    paths = Paths(data_dir=data_dir, checkpoint_dir='/tmp/dfa_checkpoints')
    tokenizer = Tokenizer(symbols)
    model = Aligner.from_checkpoint(checkpoint).eval()
    print(f'model step {model.get_step()}')
    batch_size = 4
    dataloader = new_dataloader(dataset_path=paths.data_dir / 'dataset.pkl', mel_dir=paths.mel_dir,
                                token_dir=paths.token_dir, batch_size=batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        tokens, mel, tokens_len, mel_len = to_device(batch, device)
        pred_batch = model(mel)
        for b in range(batch_size):
            this_mel_len = mel_len[b]
            this_tokens_len = tokens_len[b]
            pred = pred_batch[b, :this_mel_len, :]
            pred = torch.softmax(pred, dim=-1)
            pred = pred.detach().cpu().numpy()
            target = tokens[b, :this_tokens_len].detach().cpu().numpy()
            text = tokenizer.decode(target)
            target_len = target.shape[0]
            pred_len = pred.shape[0]
            pred_max = np.zeros((pred_len, target_len))
            for j in range(pred_len):
                pred_max[j] = pred[j, target]
            durations = extract_durations_with_dijkstra(target, pred_max)
            item_id = batch['item_id'][b]
            np.save(f'/tmp/durations/{item_id}.npy', durations)
            expanded_string = ''.join([text[i] * dur for i, dur in enumerate(list(durations))])
            #print(item_id)
            #print(text)
            #print(expanded_string)
            #print(durations)


