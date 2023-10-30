#poetry run python phospholingo visualize logs/vis_mar23_jan23_Ramasamy22_non_peptide_filtered_ST_onehot/20230303_154805/checkpoints/epoch\=5-step\=6893.ckpt data/jan23/jan23_Giansanti15_peptide_filtered_multiprotease/test.fasta shuffling_onehot_tmp.txt shuffling_onehot_tmp.svg
import time
from datetime import datetime
import math
import torch
from torch.utils.data.dataloader import DataLoader
from input_tokenizers import TokenAlphabet
import input_reader as ir
import utils
from lightning_module import LightningModule
import numpy as np
from pandas import DataFrame
import logomaker
from captum.attr import (
    DeepLiftShap,
    configure_interpretable_embedding_layer,
    InterpretableEmbeddingBase
)
from matplotlib import pyplot as plt
import pickle
from torch import nn


class PLM_wrapper_for_visualization(nn.Module):
    def __init__(self, language_model):
        """
        This is a wrapper class to be able to run DeepLiftSHAP from the Captum package on the PLM-based models. It
        first pushes the input through a one-hot encoded input layer, after which this encoding is then used to fetch
        the first-layer representation of the language model. When implemented like this, it allows to run DeepLiftSHAP
        on the one-hot encoded amino acid composition, instead of the tokens (which does not work).
        """
        super(PLM_wrapper_for_visualization, self).__init__()
        self.lm = language_model
        self.num_tokens, self.output_emb_size = self.lm.shared.weight.shape
        self.device = self.lm.device
        onehot_encoding = torch.zeros(self.num_tokens, self.num_tokens)
        for i in range(self.num_tokens):
            onehot_encoding[i][i] = 1
        self.onehot_encoder = torch.nn.Embedding.from_pretrained(
            onehot_encoding, freeze=True
        )
        self.new_first_embedding_layer = nn.Linear(self.num_tokens, self.output_emb_size, bias=False)
        self.new_first_embedding_layer.weight.data = self.lm.shared.weight.data.T
        self.lm.shared = self.new_first_embedding_layer
        self.lm.encoder.embed_tokens = self.new_first_embedding_layer

    def forward(self, input_ids, attention_mask):
        if input_ids.dtype in (torch.int64, torch.int32):
            onehot_encoded_inputs = self.onehot_encoder(input_ids)
        else:
            onehot_encoded_inputs = input_ids
        embedded_inputs = self.new_first_embedding_layer(onehot_encoded_inputs)
        return self.lm(inputs_embeds = embedded_inputs, attention_mask = attention_mask)


def run_visualize(model_loc: str, dataset_fasta: str, output_txt: str, output_img: str) -> None:
    """
    Runs predictions on a FASTA file using an already existing model, and visualizes important features via SHAP values

    Parameters
    ----------
    model : str
        The file location of an already trained model

    dataset_fasta : str
        The location of the FASTA file for which predictions are to be visualized. Predicted residues should be succeeded
        in the file by either a '#' or '@' symbol

    output_file : str
        The output txt to which predictions and feature importance scores are to be written

    output_img : str
        The output image file to which a visualization of normalized average scores will be saved
    """

    NUM_SHUFFLES = 30
    print(f'Running with {NUM_SHUFFLES} shuffles')
    model_d = torch.load(model_loc)
    config = model_d['hyper_parameters']['config']
    tokenizer = model_d['hyper_parameters']['tokenizer']
    model = LightningModule(config, 0, tokenizer)
    model.load_state_dict(model_d['state_dict'])
    model.to('cuda')
    model.eval()

    test_set = ir.SingleFastaDataset(dataset_loc=dataset_fasta, tokenizer=model.tokenizer)
    gpu_batch_size = 2#utils.get_gpu_max_batchsize(config['representation'], True)
    test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

    if config['representation'] == 'onehot':
        interpretable_emb = configure_interpretable_embedding_layer(model, 'model.encoding')
        wrapped_language_model = None
    else:
        wrapped_language_model = PLM_wrapper_for_visualization(model.model.encoding.embedding_model).to(model.device)
        model.model.encoding.embedding_model = wrapped_language_model
        interpretable_emb = configure_interpretable_embedding_layer(model, 'model.encoding.embedding_model.new_first_embedding_layer')

    shap = DeepLiftShap(model, multiply_by_inputs=True)
    with open(output_txt,'w') as write_to:
        for k,v in config.items():
            print(f'# {k}: {v}', file=write_to)
        print(f'# visualizations_on_dataset: {dataset_fasta}')

    for i,batch in enumerate(test_loader):
        print(f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} # Starting batch {i} of {len(test_loader)}')
        i+=1
        batch_results = visualize_batch(NUM_SHUFFLES, i, model, wrapped_language_model, interpretable_emb, shap, batch, config['representation'], gpu_batch_size, tokenizer)
        with open(output_txt, 'a') as write_to:
            for line in batch_results:
                print(line,file=write_to)

        ########
        # print('IF YOU SEE THIS YOU SHOULD REMOVE SOME DEBUGGIN CODE')
        # if i == 2:
        #     break
        ########

    make_average_SHAP_logo(output_img, output_txt)

def visualize_batch(NUM_SHUFFLES, batchnum, model: torch.nn.Module, wrapped_language_model: PLM_wrapper_for_visualization, interpretable_emb: InterpretableEmbeddingBase, shap: DeepLiftShap, batch: dict[str:torch.tensor], representation:str, gpu_batch_size:int, tokenizer: TokenAlphabet) -> list[str]:
    """
    Generates SHAP values for an individual batch. In contrast to training or predicting, inputs are converted to a per-
     target input, instead of having multiple targets within one input sequence. Therefore, the batch will be split up
     further into smaller mini-batches, in to not exceed the GPU capacity.

    Parameters
    ----------
    model : torch.nn.Module
        The pretrained prediction model

    interpretable_model : torch.nn.Module
        The pretrained prediction model, where an interpretable layer is inserted via the Captum package

    shap : DeepLiftShap
        An instance of the DeepLiftShap class from the Captum package

    batch : dict[str:torch.tensor]
        The batch to be processed in this function call

    representation : str
        The protein representation strategy name

    gpu_batch_size : int
        The maximum batch size to be loaded onto the GPU

    tokenizer : TokenAlphabet
        The tokenizer used to convert a string representation of a sequence to tokens

    Returns
    -------
    : list[str]

    The output lines to be printed to the results file. These are stored in a list, with every three lines containing
    the following:
        (1) an information line, with the protein id, predicted probability, label, position in fragment, position in
            protein
        (2) The amino acids in the sequence, separated by commas
        (3) The SHAP value per amino acid in the sequence, separated by commas
    """
    l = []

    prot_ids_per_seq = batch["prot_id"]
    tokens_per_seq = batch["prot_token_ids"].to(model.device)
    offsets_per_seq = batch["prot_offsets"].to(model.device)
    mask_per_seq = batch["prot_input_mask"].to(model.device)
    mask_no_extra_per_seq = batch["prot_input_mask_without_added_tokens"].to(model.device)
    targets_per_seq = batch["targets"].to(model.device)

    seqlen = targets_per_seq.shape[-1]
    bs = len(tokens_per_seq)

    # in contrast to training, visualization requires all batches to be converted to per_target inputs instead of
    # per_input targets. Therefore, extra mini-batches within regular batches are necessary
    # gather the protein ids for all targets
    protein_idx_per_tar = (
        torch.tensor(range(bs), dtype=torch.int32)
            .to(model.device)
            .expand(targets_per_seq.shape[::-1])
            .T[targets_per_seq != -1]
    )
    prot_id_per_target = [prot_ids_per_seq[id_index] for id_index in protein_idx_per_tar]
    offsets_per_target = offsets_per_seq.unsqueeze(1).repeat(1, seqlen)[targets_per_seq != -1]
    tokens_per_target = tokens_per_seq.unsqueeze(1).repeat(1, seqlen, 1)[targets_per_seq != -1]
    masks_per_target = mask_per_seq.unsqueeze(1).repeat(1, seqlen, 1)[targets_per_seq != -1]
    masks_no_extra_per_target = mask_no_extra_per_seq.unsqueeze(1).repeat(1, seqlen, 1)[targets_per_seq != -1]
    targets_per_target = targets_per_seq[targets_per_seq != -1]
    pos_in_fragment_per_target = torch.tensor(range(seqlen), dtype=torch.int32).to(model.device).expand((bs, seqlen))[targets_per_seq != -1]
    pos_in_protein_per_target = pos_in_fragment_per_target + offsets_per_target

    onehot_encoded_targets = torch.zeros(bs, seqlen, seqlen, device=model.device)
    for i in range(seqlen):
        onehot_encoded_targets[:,i,i] = 1
    onehot_encoded_targets = onehot_encoded_targets[targets_per_seq!=-1]

    all_results = []
    # divide further into mini-batches
    num_targets = len(onehot_encoded_targets)
    for n in range(math.ceil(num_targets/gpu_batch_size)):
        fro,to = n*gpu_batch_size, (n+1)*gpu_batch_size
        mb_ids_per_target = prot_id_per_target[fro:to]
        mb_offsets_per_target = offsets_per_target[fro:to]
        mb_tokens_per_target = tokens_per_target[fro:to]
        mb_masks_per_target = masks_per_target[fro:to]
        mb_masks_no_extra_per_target = masks_no_extra_per_target[fro:to]
        mb_onehot_encoded_targets = onehot_encoded_targets[fro:to]
        mb_targets = targets_per_target[fro:to]
        mb_position_per_target = pos_in_protein_per_target[fro:to]

        if len(mb_ids_per_target) == 1:
            # This is the case if we had a mini-batch size of 1 at the end of the loop - I did this workaround
            # because Captum would crash otherwise
            keep_only_one_result = True
            mb_ids_per_target = mb_ids_per_target * 2
            mb_offsets_per_target = mb_offsets_per_target.repeat_interleave(2,dim=0)
            mb_tokens_per_target = mb_tokens_per_target.repeat_interleave(2,dim=0)
            mb_masks_per_target = mb_masks_per_target.repeat_interleave(2,dim=0)
            mb_masks_no_extra_per_target = mb_masks_no_extra_per_target.repeat_interleave(2,dim=0)
            mb_onehot_encoded_targets = mb_onehot_encoded_targets.repeat_interleave(2,dim=0)
            mb_targets = mb_targets.repeat_interleave(2,dim=0)
            mb_position_per_target = mb_position_per_target.repeat_interleave(2,dim=0)
        else:
            keep_only_one_result = False

        if representation == 'onehot':
            mb_input_emb = interpretable_emb.indices_to_embeddings(
                mb_tokens_per_target,
                mb_masks_per_target,
                mb_masks_no_extra_per_target
            )
            mb_forward_output = model(mb_input_emb,
                               None,  # ignored
                               None,  # ignored
                               mb_onehot_encoded_targets)
        else:
            encoding_per_target = wrapped_language_model.onehot_encoder(mb_tokens_per_target.to(model.device))
            mb_input_emb = interpretable_emb.indices_to_embeddings(encoding_per_target)
            mb_forward_output = model(mb_input_emb.to(model.device), mb_masks_per_target.to(model.device), mb_masks_no_extra_per_target.to(model.device), mb_onehot_encoded_targets.to(model.device))

        all_res = []
        # mb_baseline = mb_input_emb.clone().to(model.device)
        import time
        for shuffle_num in range(NUM_SHUFFLES):
            t1 = time.time()
            idx = torch.randperm(mb_input_emb.size(1))
            mb_baseline = mb_input_emb.clone().to(model.device)
            mb_baseline = mb_baseline[:, idx]
            # print(idx.shape,mb_baseline.shape)
            t2 = time.time()
            mb_attribution = shap.attribute(inputs=mb_input_emb, additional_forward_args=(mb_masks_per_target, mb_masks_no_extra_per_target, mb_onehot_encoded_targets), baselines=mb_baseline)
            t3 = time.time()
            mb_results = torch.sum(mb_attribution, dim=2, keepdim=False)
            # print(f"Shuffle {shuffle_num} took {t3-t1} seconds, of which {t2-t1} seconds were spent on shuffling and {t3-t2} seconds on attribution.")
            all_res.append(mb_results)
        avg_results = torch.mean(torch.stack(all_res).transpose(2,0).transpose(1,0), dim=2)
        ##################
        # mb_d = {
        #     'mb_ids_per_target': mb_ids_per_target,
        #     'mb_offsets_per_target': mb_offsets_per_target.detach().cpu().numpy(),
        #     'mb_position_per_target': mb_position_per_target.detach().cpu().numpy(),
        #     'mb_targets': mb_targets.detach().cpu().numpy(),
        #     'mb_tokens_per_target': mb_tokens_per_target.detach().cpu().numpy(),
        #     'mb_input_emb': mb_input_emb.detach().cpu().numpy(),
        #     'avg_results': avg_results.detach().cpu().numpy(),
        # }
        # l.append(mb_d)
        ##################

        for idx in range(len(avg_results)):
            fa = model.tokenizer.get_num_tokens_added_front()
            prot_id = mb_ids_per_target[idx]
            mask = mb_masks_per_target[idx][fa:]
            ln = sum(mask)
            tokens = mb_tokens_per_target[idx][fa:]
            offset = mb_offsets_per_target[idx]
            position_in_seq = mb_position_per_target[idx]
            position_in_chunk = position_in_seq - offset
            pred = torch.sigmoid(mb_forward_output[idx])
            target = mb_targets[idx]
            scores1 = avg_results[idx]
            all_results.append(f'>{prot_id},{pred:.3f},{target},pos={position_in_chunk},actual_pos={position_in_seq}')
            all_results.append(','.join([tokenizer.all_tokens[x] if i != position_in_chunk else tokenizer.all_tokens[
                                                                                       x] + '#' + str(int(target))
                            for i, x in enumerate(tokens[:ln])]))
            all_results.append(','.join(f'{s:.3f}' for s in scores1[:ln]))
            if keep_only_one_result:
                # this is the case if we had a mini-batch size of 1 at the end of the loop - I did this workaround
                # because Captum would crash otherwise
                break
    # pickle.dump(l, open(f'shuffling_prottrans_testing_{batchnum}_numshuf{NUM_SHUFFLES}.pkl', 'wb'))
    return all_results

def make_average_SHAP_logo(output_file:str, scores_file:str, fl:int = 10) -> None:
    """
    Creates a sequence logo, after normalizing and aligning all computed SHAP values around the P-sites

    Parameters
    ----------
    output_file : str
        The file location to which to store the sequence logo image

    scores_file : str
        The file to which all calculated SHAP values were written

    fl : int
        The flanking region at each site of the P-site to show in the logo
    """
    records = []
    s = 'ACDEFGHIKLMNPQRSTVWY'
    all_results = open(scores_file).readlines()
    start_from = 0
    while all_results[start_from].startswith('#'):
        start_from+=1
    all_results = all_results[start_from:]
    for i in range(0, len(all_results), 3):
        records.append([l.strip() for l in all_results[i:i + 3]])

    # normalize
    sum_abs_vals = [sum(abs(float(x)) for x in rec[2].rstrip().split(',')) for rec in records]
    norm_coeff = 100 / (sum(sum_abs_vals) / len(sum_abs_vals))

    # create image
    totals = np.zeros((len(s), 2 * fl + 1))
    counts = np.zeros((len(s), 2 * fl + 1))
    for rec in records:
        seq = rec[1].rstrip(',').split(',')
        scores = [float(x) * norm_coeff for x in rec[2].rstrip().split(',')]

        for pos in range(len(seq)):
            if '#' in seq[pos]: # denoted as for instance "T#1" or "S#0"
                for j in range(fl * 2 + 1):
                    if 0 <= pos - fl + j < len(seq):
                        aa = seq[pos - fl + j][:1]  # cut off #0 and #1
                        if aa in s:
                            scr = scores[pos - fl + j]
                            if aa in s:
                                aa_index = s.index(aa)
                                totals[aa_index][j] += scr
                                counts[aa_index][j] += 1

    counts[counts == 0] = 1 # to avoid division by zero
    averages = (totals / counts)
    df = DataFrame(averages.transpose(), columns=list(s))
    fig, ax = plt.subplots(1, 1, figsize=[4, 2])
    logo = logomaker.Logo(df, color_scheme='chemistry', ax=ax)
    ticks = list(range(0, fl * 2 + 1))
    labs = [f'-{i}' for i in range(1, fl + 1)][::-1] + ['P'] + [f'+{i}' for i in range(1, fl + 1)]
    plt.xticks(ticks[::2], labs[::2], fontsize=9)
    plt.ylabel('importance score', labelpad=0)
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file)