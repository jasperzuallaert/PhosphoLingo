import input_reader as ir
import torch.nn
from torch.utils.data.dataloader import DataLoader
from lightning_module import LightningModule
import utils

def run_predict(model_loc: str, dataset_fasta: str, output_file: str) -> None:
    """
    Runs predictions on a FASTA file using an already existing model

    Parameters
    ----------
    model : str
        The file location of an already trained model

    dataset_fasta : str
        The location of the FASTA file for which predictions are to be made. Predicted residues should be succeeded in
        the file by either a '#' or '@' symbol

    output_file : str
        The output csv file to which predictions are written
    """
    model_d = torch.load(model_loc)
    config = model_d['hyper_parameters']['config']
    model = LightningModule(config, 0, model_d['hyper_parameters']['tokenizer'])
    model.load_state_dict(model_d['state_dict'])
    model.eval()
    test_set = ir.SingleFastaDataset(dataset_loc=dataset_fasta, tokenizer=model.tokenizer)
    gpu_batch_size = utils.get_gpu_max_batchsize(config['representation'], True)
    test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

    with open(output_file, 'w') as write_to:
        print('prot_id,position,pred', file=write_to)
        with torch.no_grad():
            for batch in test_loader:
                (
                    prot_ids,
                    site_positions,
                    logit_outputs,
                    predicted_probs,
                    targets,
                ) = model.process_batch(batch)

                for id, pos, prob in zip(prot_ids, site_positions, predicted_probs):
                    print(','.join([id, str(int(pos) + 1), '{:.3f}'.format(float(prob))]), file=write_to)
