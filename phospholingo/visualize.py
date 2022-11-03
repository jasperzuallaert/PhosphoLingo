import itertools
import math
import torch
import numpy as np

from captum.attr import (
    DeepLiftShap,
    configure_interpretable_embedding_layer,
)

import input_reader as ir
import torch.nn
from torch.utils.data.dataloader import DataLoader
import utils

def run_visualize(model_loc: str, dataset_fasta: str, output_file: str) -> None:
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
    """
    model = torch.load(model_loc)
    test_set = ir.SingleFastaDataset(dataset_loc=dataset_fasta, tokenizer=model.tokenizer)
    gpu_batch_size = utils.get_gpu_max_batchsize()
    test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

    model.eval()
    interpretable_model = configure_interpretable_embedding_layer(model, 'encoding')

    shap = DeepLiftShap(model, multiply_by_inputs=True)
    with open(output_file, 'w') as write_to:
        for batch in test_loader:
            l_to_print = visualize_batch(interpretable_model, model, shap, batch)





            for id, pos, prob in zip(prot_ids, site_positions, predicted_probs):
                print(','.join([id, str(int(pos) + 1), '{:.3f}'.format(float(prob))]), file=write_to)

def visualize_batch(interpretable_model, model, shap, batch):
    batch_prot_ids = batch['prot_id']
    batch_prot_seq_tokens = batch['prot_token_ids'].to(model.device)
    batch_prot_offsets = batch['prot_offsets']
    batch_prot_input_masks = batch['prot_input_mask'].to(model.device)
    batch_prot_input_masks_no_extra = batch['prot_input_mask_without_added_tokens'].to(model.device)
    batch_targets = batch['targets'].to(model.device)

    seqlen = batch_targets.shape[-1]
    bs = len(batch_prot_ids)

    batch_prot_tok_per_target = batch_prot_seq_tokens.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
    batch_prot_input_masks_per_target = batch_prot_input_masks.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
    batch_prot_input_masks_no_extra_per_target = batch_prot_input_masks_no_extra.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]

    target_pos_candidates = torch.zeros(bs, seqlen, seqlen, device=model.device)
    for i in range(seqlen):
        target_pos_candidates[:,i,i] = 1
    batch_target_positions_per_target = target_pos_candidates[batch_targets!=-1]

    input_emb = interpretable_model.indices_to_embeddings(batch_prot_tok_per_target, batch_prot_input_masks_per_target,
                                                          batch_prot_input_masks_no_extra_per_target)

    forward_output = model(input_emb,
                           None,  # ignored
                           None,  # ignored
                           batch_target_positions_per_target)

    # baseline
    seqlens = torch.sum(batch_prot_input_masks_no_extra_per_target, dim=-1, keepdim=True)  # (bs, 1) -> with lengths
    mask = batch_prot_input_masks_no_extra_per_target[:, model.tokenizer.get_num_tokens_added_front():]
    masked_mean_per_sequence = torch.sum(input_emb * mask.unsqueeze(-1), dim=1) / seqlens
    baseline = mask.unsqueeze(-1) * masked_mean_per_sequence.unsqueeze(1)

    attribution = shap.attribute(inputs=input_emb, additional_forward_args=(batch_prot_tok_per_target, batch_prot_input_masks_no_extra_per_target, batch_target_positions_per_target), baselines=baseline)
    results = torch.sum(attribution, dim=2, keepdim=False)

    for idx in range(len(batch_prot_ids)):
        fa = model.tokenizer.get_num_tokens_added_front()
        prot_id = batch_prot_ids[idx]
        mask = batch_prot_input_masks_per_target[idx][fa:]
        ln = sum(mask)
        tokens = part_tok_per_target[idx][fa:]
        offset = part_offsets[idx]
        position_in_seq = part_act_pos[idx]
        position_in_chunk = position_in_seq - offset
        pred = probabilities[idx]
        target = part_targets[idx]
        scores1 = results[idx]
        print(f'>{prot_id},{pred:.3f},{target},pos={position_in_chunk},actual_pos={position_in_seq}',
              file=write_to)
        print(','.join([tokenizer.all_tokens[x] if i != position_in_chunk else tokenizer.all_tokens[
                                                                                   x] + '#' + str(int(target))
                        for i, x in enumerate(tokens[:ln])]), file=write_to)
        print(','.join(f'{s:.3f}' for s in scores1[:ln]), file=write_to)



    return results

def visualize_shap(exp_name, model, test_loader, representation, device, part=0, num_parts=1):
    for batch in test_loader:
        batch_prot_ids = batch['prot_id']
        print(batch_prot_ids)
        batch_prot_seq_tokens = batch['prot_token_ids'].to(device)
        batch_prot_offsets = batch['prot_offsets']
        batch_prot_input_masks = batch['prot_input_mask'].to(device)
        batch_prot_input_masks_no_extra = batch['prot_input_mask_without_added_tokens'].to(device)
        batch_targets = batch['targets'].to(device)

        '''if BASELINE_IS_AVERAGE_EMBEDDING:
            full_batch_unique_input_embeddings = interpretable_model.indices_to_embeddings(batch_prot_seq_tokens, batch_prot_input_masks, batch_prot_input_masks_no_extra)
            seqlens = torch.sum(batch_prot_input_masks_no_extra, dim=-1, keepdim=True)  # (bs, 1) -> with lengths
            mask = batch_prot_input_masks_no_extra[:,tokenizer.get_num_tokens_added_front():]
            masked_mean_per_sequence = torch.sum(full_batch_unique_input_embeddings * mask.unsqueeze(-1), dim=1) / seqlens
            # unsqueeze gives bs, 512, 1
            # * gives (bs, 512, 768) * (bs, 512, 1) = (bs, 512, 768)
            # sum gives bs, 768
            # / gives (bs, 768) / (bs, 1) = (bs, 768)
            batch_mean = torch.mean(masked_mean_per_sequence, dim=0)
            # gives (768)'''
        seqlen = batch_targets.shape[-1]
        bs = len(batch_prot_ids)

        batch_prot_tok_per_target = batch_prot_seq_tokens.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
        batch_prot_input_masks_per_target = batch_prot_input_masks.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
        batch_prot_input_masks_no_extra_per_target = batch_prot_input_masks_no_extra.unsqueeze(1).repeat(1, seqlen, 1)[batch_targets != -1]
        num_targets = len(batch_prot_tok_per_target)

        target_pos_candidates = torch.zeros(bs, seqlen, seqlen, device=device)
        for i in range(seqlen):
            target_pos_candidates[:,i,i] = 1
        batch_target_positions_per_target = target_pos_candidates[batch_targets!=-1]

        ### all of this was already per-target
        # gather the offsets for all targets
        considered_prot_offsets = batch_prot_offsets.expand(batch_targets.shape[::-1]).T[batch_targets != -1].to(device)

        # gather the positions within the fragment for each target
        considered_positions_in_fragment = torch.tensor(range(seqlen), dtype=torch.int32).to(device) \
            .expand((bs, seqlen))[batch_targets != -1]

        batch_prot_id_indices = torch.tensor(range(bs), dtype=torch.int32).to(device) \
            .expand(batch_targets.shape[::-1]) \
            .T[batch_targets != -1]

        considered_prot_ids = [batch_prot_ids[id_index] for id_index in batch_prot_id_indices]
        # calculate the actual positions for each target, by adding up the position + the offset
        considered_actual_positions = considered_positions_in_fragment + considered_prot_offsets
        # gather the annotations for all targets
        considered_targets = batch_targets[batch_targets != -1]
        # gather the predicted probabilities for all targets

        for partN in range(math.ceil(num_targets/MAX_BATCH_SIZE)):
            part_tok_per_target = batch_prot_tok_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
            if part_tok_per_target.shape[0] == 1: continue # TEMPORARY SKIP TO AVOID BUG...
            part_msk_per_target = batch_prot_input_masks_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
            part_msk_no_ex_per_target = batch_prot_input_masks_no_extra_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
            part_target_pos_per_target = batch_target_positions_per_target[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]

            part_prot_ids = considered_prot_ids[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
            part_offsets = considered_prot_offsets[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
            part_act_pos = considered_actual_positions[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]
            part_targets = considered_targets[partN*MAX_BATCH_SIZE:(partN+1)*MAX_BATCH_SIZE]

            if METHOD_IS_SHAP or not METHOD_IS_OCCLUSION:
                input_emb = interpretable_model.indices_to_embeddings(part_tok_per_target, part_msk_per_target, part_msk_no_ex_per_target)
                forward_output = model(input_emb,
                                        part_msk_per_target, # ignored
                                        part_msk_no_ex_per_target, # ignored
                                        part_target_pos_per_target)
            else:
                forward_output = model(part_tok_per_target,
                                        part_msk_per_target, # ignored
                                        part_msk_no_ex_per_target, # ignored
                                        part_target_pos_per_target)
            probabilities = torch.sigmoid(forward_output)

            if (METHOD_IS_SHAP and BASELINE_IS_AVERAGE_EMBEDDING) or not METHOD_IS_OCCLUSION:
                ### NEWTRY
                # full_batch_unique_input_embeddings = interpretable_model.indices_to_embeddings(batch_prot_seq_tokens, batch_prot_input_masks, batch_prot_input_masks_no_extra)
                seqlens = torch.sum(part_msk_no_ex_per_target, dim=-1, keepdim=True)  # (bs, 1) -> with lengths
                mask = part_msk_no_ex_per_target[:,tokenizer.get_num_tokens_added_front():]
                masked_mean_per_sequence = torch.sum(input_emb * mask.unsqueeze(-1), dim=1) / seqlens
                # unsqueeze gives bs, 512, 1
                # * gives (bs, 512, 768) * (bs, 512, 1) = (bs, 512, 768)
                # sum gives bs, 768
                # / gives (bs, 768) / (bs, 1) = (bs, 768)
                # batch_mean = torch.mean(masked_mean_per_sequence, dim=0)
                # gives (768)
                ###/NEWTRY
                baseline = mask.unsqueeze(-1) * masked_mean_per_sequence.unsqueeze(1)
            elif METHOD_IS_OCCLUSION:
                baseline = torch.zeros_like(part_tok_per_target, device=device)
            else:
                baseline = torch.zeros_like(input_emb, device=device)

            # print('input_emb.shape',input_emb.shape)
            # print('part_msk_per_target.shape',part_msk_per_target.shape)
            # print('part_msk_no_ex_per_target.shape',part_msk_no_ex_per_target.shape)
            # print('part_target_pos_per_target.shape',part_target_pos_per_target.shape)
            # print('baseline.shape',baseline.shape)
            # print()
            if METHOD_IS_SHAP:
                # ig = GradientShap(model, multiply_by_inputs=True)
                ig = DeepLiftShap(model, multiply_by_inputs=True)
                attribution = ig.attribute(inputs=input_emb,additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target),baselines=baseline)
            elif METHOD_IS_OCCLUSION:
                ig = Occlusion(model)
                attribution = ig.attribute(inputs=part_tok_per_target,additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target), sliding_window_shapes=(1,), baselines=unk_pos)
            else:
                ig = IntegratedGradients(model, multiply_by_inputs=True)
                attribution = ig.attribute(inputs=input_emb,additional_forward_args=(part_msk_per_target,part_msk_no_ex_per_target,part_target_pos_per_target),n_steps=20,baselines=baseline)

            if METHOD_IS_OCCLUSION:
                results = attribution
            else:
                results = torch.sum(attribution, dim=2, keepdim=False)


            write_to = open(output_filename, 'a')
            for idx in range(len(part_prot_ids)):
                fa = tokenizer.get_num_tokens_added_front()
                prot_id = part_prot_ids[idx]
                mask = part_msk_per_target[idx][fa:]
                ln = sum(mask)
                tokens = part_tok_per_target[idx][fa:]
                offset = part_offsets[idx]
                position_in_seq = part_act_pos[idx]
                position_in_chunk = position_in_seq - offset
                pred = probabilities[idx]
                target = part_targets[idx]
                scores1 = results[idx]
                print(f'>{prot_id},{pred:.3f},{target},pos={position_in_chunk},actual_pos={position_in_seq}',
                      file=write_to)
                print(','.join([tokenizer.all_tokens[x] if i != position_in_chunk else tokenizer.all_tokens[
                                                                                           x] + '#' + str(int(target))
                                for i, x in enumerate(tokens[:ln])]), file=write_to)
                print(','.join(f'{s:.3f}' for s in scores1[:ln]), file=write_to)
            write_to.close()

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