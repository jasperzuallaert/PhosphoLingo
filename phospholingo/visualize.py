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
    model_d = torch.load(model_loc)
    config = model_d['hyper_parameters']['config']
    tokenizer = model_d['hyper_parameters']['tokenizer']
    model = LightningModule(config, tokenizer)
    model.load_state_dict(model_d['state_dict'])
    model.eval()
    test_set = ir.SingleFastaDataset(dataset_loc=dataset_fasta, tokenizer=model.tokenizer)
    gpu_batch_size = utils.get_gpu_max_batchsize(config['representation'], True)
    test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

    interpretable_emb = configure_interpretable_embedding_layer(model, 'model.encoding')

    shap = DeepLiftShap(model, multiply_by_inputs=True)
    i=0
    open(output_txt,'w')
    for batch in test_loader:
        print(f'batch {i}')
        i+=1
        batch_results = visualize_batch(model, interpretable_emb, shap, batch, config['representation'], gpu_batch_size, tokenizer)
        with open(output_txt, 'a') as write_to:
            for line in batch_results:
                print(line,file=write_to)

    make_average_SHAP_logo(output_img, output_txt)

def visualize_batch(model: torch.nn.Module, interpretable_emb: InterpretableEmbeddingBase, shap: DeepLiftShap, batch: dict[str:torch.tensor], representation:str, gpu_batch_size:int, tokenizer: TokenAlphabet) -> list[str]:
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

    prot_ids_per_seq = batch["prot_id"]
    tokens_per_seq = batch["prot_token_ids"].to(model.device)
    offsets_per_seq = batch["prot_offsets"]
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
    print(tokens_per_target.shape)
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

        # A quick fix that occasionally skips a single case, when the mini-batch size of 1 might be found at the end of
        # a batch. This throws an AssertionError (though it should be supported). To be fixed in a later iteration
        if len(mb_ids_per_target) == 1:
            print(f'Skipped input of size {mb_tokens_per_target.shape} due to Captum API issue')
            continue

        mb_input_emb = interpretable_emb.indices_to_embeddings(
            mb_tokens_per_target,
            mb_masks_per_target,
            mb_masks_no_extra_per_target
        )

        mb_forward_output = model(mb_input_emb,
                               None,  # ignored
                               None,  # ignored
                               mb_onehot_encoded_targets)

        if representation == 'onehot':
            # if representation is one-hot, an all-zero baseline is used
            mb_baseline = torch.zeros_like(mb_input_emb).to(model.device)
        else:
            # if language model, construct the baseline for each protein fragment as the average embedding for that fragment
            mb_seqlens = torch.sum(mb_masks_no_extra_per_target, dim=-1, keepdim=True)  # (bs, 1) -> with lengths
            mb_mask = mb_masks_no_extra_per_target[:, model.tokenizer.get_num_tokens_added_front():]
            mb_masked_mean_per_sequence = torch.sum(mb_input_emb * mb_mask.unsqueeze(-1), dim=1) / mb_seqlens
            mb_baseline = mb_mask.unsqueeze(-1) * mb_masked_mean_per_sequence.unsqueeze(1)

        mb_attribution = shap.attribute(inputs=mb_input_emb, additional_forward_args=(mb_masks_per_target, mb_masks_no_extra_per_target, mb_onehot_encoded_targets), baselines=mb_baseline)
        mb_results = torch.sum(mb_attribution, dim=2, keepdim=False)

        # print('a',mb_tokens_per_target[0])
        # print('b',mb_onehot_encoded_targets[0])
        # print('c',mb_offsets_per_target[0])
        # print('d',mb_masks_per_target[0])
        # print('e',mb_input_emb[0][0])
        # print('e',mb_input_emb[0][1])
        # print('e',mb_input_emb[0][2])
        # print('f',mb_targets[0])
        # print('g',mb_position_per_target[0])
        #
        # print('h1',mb_attribution[0])
        # print('h2',mb_attribution[0][0])
        # print('h3',mb_attribution[0][1])
        # print('h4',mb_attribution[0][2])
        # print('i',mb_results[0])
        #
        # print('j',mb_baseline[0])

        for idx in range(len(mb_results)):
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
            scores1 = mb_results[idx]
            all_results.append(f'>{prot_id},{pred:.3f},{target},pos={position_in_chunk},actual_pos={position_in_seq}')
            all_results.append(','.join([tokenizer.all_tokens[x] if i != position_in_chunk else tokenizer.all_tokens[
                                                                                       x] + '#' + str(int(target))
                            for i, x in enumerate(tokens[:ln])]))
            all_results.append(','.join(f'{s:.3f}' for s in scores1[:ln]))
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
            if '#' in seq[pos]: #denoted as for instance "T#1" or "S#0"
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

# def visualize_shap(exp_name, model, test_loader, representation, device, part=0, num_parts=1):
#     for batch in test_loader:
#         batch_prot_ids = batch['prot_id']
#         print(batch_prot_ids)
#         batch_prot_seq_tokens = batch['prot_token_ids'].to(device)
#         batch_prot_offsets = batch['prot_offsets']
#         batch_prot_input_masks = batch['prot_input_mask'].to(device)
#         batch_prot_input_masks_no_extra = batch['prot_input_mask_without_added_tokens'].to(device)
#         batch_targets = batch['targets'].to(device)
#
#         '''if BASELINE_IS_AVERAGE_EMBEDDING:
#             full_batch_unique_input_embeddings = interpretable_model.indices_to_embeddings(batch_prot_seq_tokens, batch_prot_input_masks, batch_prot_input_masks_no_extra)
#             seqlens = torch.sum(batch_prot_input_masks_no_extra, dim=-1, keepdim=True)  # (bs, 1) -> with lengths
#             mask = batch_prot_input_masks_no_extra[:,tokenizer.get_num_tokens_added_front():]
#             masked_mean_per_sequence = torch.sum(full_batch_unique_input_embeddings * mask.unsqueeze(-1), dim=1) / seqlens
#             # unsqueeze gives bs, 512, 1
#             # * gives (bs, 512, 768) * (bs, 512, 1) = (bs, 512, 768)
#             # sum gives bs, 768
#             # / gives (bs, 768) / (bs, 1) = (bs, 768)
#             batch_mean = torch.mean(masked_mean_per_sequence, dim=0)
#             # gives (768)'''
#         seqlen = batch_targets.shape[-1]
#         bs = len(batch_prot_ids)
#
#         batch_prot_tok_per_target = batch_prot_seq_tokens.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
#         batch_prot_input_masks_per_target = batch_prot_input_masks.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
#         batch_prot_input_masks_no_extra_per_target = batch_prot_input_masks_no_extra.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
#         num_targets = len(batch_prot_input_masks_no_extra_per_target)
#
#         target_pos_candidates = torch.zeros(bs, seqlen, seqlen, device=device)
#         for i in range(seqlen):
#             target_pos_candidates[:,i,i] = 1
#         batch_target_positions_per_target = target_pos_candidates[batch_targets!=-1]
#
#         ### all of this was already per-target
#         # gather the offsets for all targets
#         considered_prot_offsets = batch_prot_offsets.expand(batch_targets.shape[::-1]).T[batch_targets != -1].to(device)
#
#         # gather the positions within the fragment for each target
#         considered_positions_in_fragment = torch.tensor(range(seqlen), dtype=torch.int32).to(device) \
#             .expand((bs, seqlen))[batch_targets != -1]
#
#         batch_prot_id_indices = torch.tensor(range(bs), dtype=torch.int32).to(device) \
#             .expand(batch_targets.shape[::-1]) \
#             .T[batch_targets != -1]
#
#         considered_prot_ids = [batch_prot_ids[id_index] for id_index in batch_prot_id_indices]
#         # calculate the actual positions for each target, by adding up the position + the offset
#         considered_actual_positions = considered_positions_in_fragment + considered_prot_offsets
#         # gather the annotations for all targets
#         considered_targets = batch_targets[batch_targets != -1]
#         # gather the predicted probabilities for all targets
#
#         for partN in range(math.ceil(num_targets/MAX_BATCH_SIZE)):
#             part_tok_per_target = batch_prot_tok_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#             if part_tok_per_target.shape[0] == 1: continue # TEMPORARY SKIP TO AVOID BUG...
#             part_msk_per_target = batch_prot_input_masks_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#             part_msk_no_ex_per_target = batch_prot_input_masks_no_extra_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#             part_target_pos_per_target = batch_target_positions_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#
#             part_prot_ids = considered_prot_ids[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#             part_offsets = considered_prot_offsets[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#             part_act_pos = considered_actual_positions[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#             part_targets = considered_targets[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
#
#             if METHOD_IS_SHAP or not METHOD_IS_OCCLUSION:
#                 input_emb = interpretable_model.indices_to_embeddings(part_tok_per_target, part_msk_per_target, part_msk_no_ex_per_target)
#                 forward_output = model(input_emb,
#                                         part_msk_per_target, # ignored
#                                         part_msk_no_ex_per_target, # ignored
#                                         part_target_pos_per_target)
#             else:
#                 forward_output = model(part_tok_per_target,
#                                         part_msk_per_target, # ignored
#                                         part_msk_no_ex_per_target, # ignored
#                                         part_target_pos_per_target)
#             probabilities = torch.sigmoid(forward_output)
#
#             if (METHOD_IS_SHAP and BASELINE_IS_AVERAGE_EMBEDDING) or not METHOD_IS_OCCLUSION:
#                 ### NEWTRY
#                 # full_batch_unique_input_embeddings = interpretable_model.indices_to_embeddings(batch_prot_seq_tokens, batch_prot_input_masks, batch_prot_input_masks_no_extra)
#                 seqlens = torch.sum(part_msk_no_ex_per_target, dim=-1, keepdim=True)  # (bs, 1) -> with lengths
#                 mask = part_msk_no_ex_per_target[:,tokenizer.get_num_tokens_added_front():]
#                 masked_mean_per_sequence = torch.sum(input_emb * mask.unsqueeze(-1), dim=1) / seqlens
#                 # unsqueeze gives bs, 512, 1
#                 # * gives (bs, 512, 768) * (bs, 512, 1) = (bs, 512, 768)
#                 # sum gives bs, 768
#                 # / gives (bs, 768) / (bs, 1) = (bs, 768)
#                 # batch_mean = torch.mean(masked_mean_per_sequence, dim=0)
#                 # gives (768)
#                 ###/NEWTRY
#                 baseline = mask.unsqueeze(-1) * masked_mean_per_sequence.unsqueeze(1)
#             elif METHOD_IS_OCCLUSION:
#                 baseline = torch.zeros_like(part_tok_per_target, device=device)
#             else:
#                 baseline = torch.zeros_like(input_emb, device=device)
#
#             # print('input_emb.shape',input_emb.shape)
#             # print('part_msk_per_target.shape',part_msk_per_target.shape)
#             # print('part_msk_no_ex_per_target.shape',part_msk_no_ex_per_target.shape)
#             # print('part_target_pos_per_target.shape',part_target_pos_per_target.shape)
#             # print('baseline.shape',baseline.shape)
#             # print()
#             if METHOD_IS_SHAP:
#                 # ig = GradientShap(model, multiply_by_inputs=True)
#                 ig = DeepLiftShap(model, multiply_by_inputs=True)
#                 attribution = ig.attribute(inputs=input_emb,additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target),baselines=baseline)
#             elif METHOD_IS_OCCLUSION:
#                 ig = Occlusion(model)
#                 attribution = ig.attribute(inputs=part_tok_per_target,additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target), sliding_window_shapes=(1,), baselines=unk_pos)
#             else:
#                 ig = IntegratedGradients(model, multiply_by_inputs=True)
#                 attribution = ig.attribute(inputs=input_emb,additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target),n_steps=20,baselines=baseline)
#
#             if METHOD_IS_OCCLUSION:
#                 results = attribution
#             else:
#                 results = torch.sum(attribution, dim=2, keepdim=False)
#
#
#             write_to = open(output_filename, 'a')
#             for idx in range(len(part_prot_ids)):
#                 fa = tokenizer.get_num_tokens_added_front()
#                 prot_id = part_prot_ids[idx]
#                 mask = part_msk_per_target[idx][fa:]
#                 ln = sum(mask)
#                 tokens = part_tok_per_target[idx][fa:]
#                 offset = part_offsets[idx]
#                 position_in_seq = part_act_pos[idx]
#                 position_in_chunk = position_in_seq - offset
#                 pred = probabilities[idx]
#                 target = part_targets[idx]
#                 scores1 = results[idx]
#                 print(f'>{prot_id},{pred:.3f},{target},pos={position_in_chunk},actual_pos={position_in_seq}',
#                       file=write_to)
#                 print(','.join([tokenizer.all_tokens[x] if i != position_in_chunk else tokenizer.all_tokens[
#                                                                                            x] + '#' + str(int(target))
#                                 for i, x in enumerate(tokens[:ln])]), file=write_to)
#                 print(','.join(f'{s:.3f}' for s in scores1[:ln]), file=write_to)
#             write_to.close()

    #         lig1 = LayerIntegratedGradients(model, pre_embedding_layer, multiply_by_inputs=True)
    #         print('reached k')
    #
    #         # set baseline: <unk> for length, <mask> for rest
    #         batch_tokens_baseline = torch.zeros_like(part_tok_per_target)
    #         batch_tokens_baseline[part_msk_per_target == 1] = tokenizer.unk_idx
    #         batch_tokens_baseline[part_msk_per_target == 0] = tokenizer.padding_idx
    #         # if there is a cls token in the alphabet, set the token at the appropriate positions of the baseline as well
    #         if tokenizer.cls_idx:
    #             batch_tokens_baseline[part_tok_per_target == tokenizer.cls_idx] = tokenizer.cls_idx
    #
    #         att1 = lig1.attribute(inputs=(part_tok_per_target), baselines=(batch_tokens_baseline), additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target), n_steps=20, internal_batch_size=MAX_BATCH_SIZE)
    #         print('reached l')
    #         # results1 = model.encoding.aggregate_attributions_per_position(batch_prot_seq_tokens, att1)
    #         if representation == 'onehot':
    #             # padding token idx should not be counted (see OneHotEncoding class in network_architectures)
    #             # so putting -1 to have all idx correct - the attribution of the padding does not matter, it is discarded
    #             # results1 = torch.gather(att1, dim=2, index=f.relu(part_tok_per_target.unsqueeze(-1)-1)).squeeze(-1)
    #             results1 = torch.sum(att1, dim=2, keepdim=False)
    #         else:
    #             results1 = torch.sum(att1, dim=2, keepdim=False)
    #         print('reached m')
    #
    #         write_to = open(output_filename,'a')
    #         for idx in range(len(part_prot_ids)):
    #             fa = tokenizer.get_num_tokens_added_front()
    #             prot_id = part_prot_ids[idx]
    #             mask = part_msk_per_target[idx][fa:]
    #             ln = sum(mask)
    #             tokens = part_tok_per_target[idx][fa:]
    #             offset = part_offsets[idx]
    #             position_in_seq = part_act_pos[idx]
    #             position_in_chunk = position_in_seq - offset
    #             pred = probabilities[idx]
    #             target = part_targets[idx]
    #             scores1 = results1[idx][fa:]
    #             print(f'>{prot_id},{pred:.3f},{target},pos={position_in_chunk},actual_pos={position_in_seq}',file=write_to)
    #             print(','.join([tokenizer.all_tokens[x] if i != position_in_chunk else tokenizer.all_tokens[x] + '#' + str(int(target)) for i, x in enumerate(tokens[:ln])]),file=write_to)
    #             print(','.join(f'{s:.3f}' for s in scores1[:ln]),file=write_to)
    #         write_to.close()
    #
    #         # cs = x.gather(2,batch_x_seqs.unsqueeze(-1)).squeeze()
    #         # all_nuc_idx.extend(batch_x_seqs)
    #         # all_cs_idx.extend(cs)


# TODO TODO ADD IMAGE MAKER?