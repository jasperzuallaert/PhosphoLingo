from math import floor
import torch
import numpy as np
from Bio import SeqIO
from input_tokenizers import TokenAlphabet
from typing import Any

# Default fragment length used throughout the publication
_PROTEIN_FRAGMENT_LENGTH = 512

class PTMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        unique_prot_ids: set[str],
        tokenizer: TokenAlphabet,
        num_pos: int,
        num_neg: int,
    ) -> None:
        """
        Dataset class for PTM annotations. Sequences in input data should be in a FASTA format, with positive
        annotations indicated by a '#' succeeding the modified residue, and negative annotations indicated by a '@'.

        Parameters
        ----------
        data : list[dict[str, Any]]
            Data entries holding sequence and annotation information for each protein fragment. Each dict entry should
            have  ``prot_id``, ``offset``, ``seq_data``, and ``labels`` information
        unique_prot_ids : set[str]
            All unique protein ids that are found in the data entries
        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation
        num_pos : int
            The number of positive annotations in the data entries
        num_neg : int
            The number of negative annotations in the data entries

        Attributes
        ----------
        data : list[dict[str, Any]]
            Stores the data in input
        tokenizer : TokenAlphabet
            Stores the tokenizer in input
        unique_prot_ids : set[str]
            Stores the unique protein ids in input
        num_pos : int
            Stores the number of positive annotations in input
        num_neg : int
            Stores the number of negative annotations in input

        """
        self.data = data
        self.tokenizer = tokenizer
        self.unique_prot_ids = unique_prot_ids
        self.num_pos = num_pos
        self.num_neg = num_neg

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        item = self.data[index]
        prot_id = item["prot_id"]
        prot_offset = item["offset"]
        prot_token_ids = self.tokenizer.tokenize(item["seq_data"])
        prot_input_mask = np.ones_like(prot_token_ids)
        prot_input_mask_no_extra = self.tokenizer.get_mask_no_extra(item["seq_data"])
        labels = np.asarray(item["labels"])
        return (
            prot_id,
            prot_offset,
            prot_token_ids,
            prot_input_mask,
            prot_input_mask_no_extra,
            labels,
        )

    def collate_fn(self, batch) -> dict[str, Any]:
        (
            prot_id,
            prot_offset,
            prot_token_ids,
            prot_input_mask,
            prot_input_mask_no_extra,
            labels,
        ) = tuple(zip(*batch))
        prot_token_ids = torch.from_numpy(
            pad_sequences(prot_token_ids, self.tokenizer.tok_to_idx['<pad>'])
        )
        prot_input_mask = torch.from_numpy(pad_sequences(prot_input_mask, 0))
        prot_input_mask_no_extra = torch.from_numpy(
            pad_sequences(prot_input_mask_no_extra, 0)
        )
        prot_offset = torch.tensor(prot_offset)
        targets = torch.from_numpy(pad_sequences(np.asarray(labels), -1))
        output = {
            "prot_id": prot_id,
            "prot_token_ids": prot_token_ids,
            "prot_offsets": prot_offset,
            "prot_input_mask": prot_input_mask,
            "prot_input_mask_without_added_tokens": prot_input_mask_no_extra,
            "targets": targets,
        }
        return output

    def print_details(self, name: str, extra_info: str = "") -> None:
        """
        Prints dataset details, such as the number of proteins, the number of positive annotations, and the number of
        negative annotations

        Parameters
        ----------
        name : str
            Dataset name to be able to make the distinction between different print calls
        extra_info : str
            Extra information to be printed
        """
        if extra_info:
            print(f"--- Loaded {name} ({extra_info}) data ---")
        else:
            print(f"--- Loaded {name} data ---")
        print(f"- Number of proteins: {len(self.unique_prot_ids)}")
        print(f"- Number of positive sites: {self.num_pos}")
        print(f"- Number of negative sites: {self.num_neg}")
        print()


class SingleFastaDataset(PTMDataset):
    def __init__(
        self, dataset_loc: str, tokenizer: TokenAlphabet, train_valid_test: str = None
    ) -> None:
        """
        Create a dataset from a single FASTA file

        Parameters
        ----------
        dataset_loc : str
            Specification of the FASTA file, located within the ``data/`` directory. Can be either a directory, in which
            train.fasta, valid.fasta or test.fasta will be used, based on the ``train_valid_test`` parameter (used in
            run_train), or a FASTA file (used in run_predict and run_visualize)

        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation

        train_valid_test : str
            Speficies whether to load the training, validation or test set. Value should be one of ``train``, ``valid``,
            or ``test``
        """
        assert train_valid_test in (None, "train", "valid", "test")
        if dataset_loc.endswith('.fasta') or dataset_loc.endswith('.fa'):
            assert train_valid_test is None
            filename = dataset_loc
        else:
            filename = f"data/{dataset_loc}/{train_valid_test}.fasta"
        fasta_dict = get_fasta_dict(filename, exclude_proteins=[])
        data, unique_prot_ids, num_pos, num_neg = _read_records(fasta_dict)
        super(SingleFastaDataset, self).__init__(
            data, unique_prot_ids, tokenizer, num_pos, num_neg
        )
        self.print_details(dataset_loc, extra_info=train_valid_test)


class MultiFoldDataset(PTMDataset):
    """
    Create a dataset from one ore more partial FASTA files (so-called folds). It is possible to exclude proteins
    that are present in for instance an evaluation set

    Parameters
    ----------
    dataset_loc : str
        Directory of the FASTA files, located within the ``data/`` directory

    tokenizer : TokenAlphabet
        The tokenizer used for the selected protein representation

    exclude_proteins : list[str]
        Protein ids to exclude from the input FASTA

    folds : set[int]
        The fold numbers to include in this Dataset
    """

    def __init__(
        self,
        dataset_loc: str,
        tokenizer: TokenAlphabet,
        exclude_proteins: list[str],
        folds: set[int],
    ) -> None:
        # gather all annotations from different folds into one dictionary
        fasta_dict = {}
        for fold_num in folds:
            fold_fasta_dict = get_fasta_dict(
                f'data/{dataset_loc.rstrip("/")}/fold{fold_num}.fasta',
                exclude_proteins=exclude_proteins,
            )
            fasta_dict.update(fold_fasta_dict)
        data, unique_prot_ids, num_pos, num_neg = _read_records(fasta_dict)
        super(MultiFoldDataset, self).__init__(
            data, unique_prot_ids, tokenizer, num_pos, num_neg
        )
        self.print_details(dataset_loc, extra_info=str(folds))

    def get_proteins_in_dataset(self):
        return self.all_prot_ids


# Gets all entries in a fasta file, while excluding proteins in a given set
def get_fasta_dict(filename: str, exclude_proteins: list[str]) -> dict[str, str]:
    """
    Reads an annotated FASTA file

    Parameters
    ----------
    filename : str
        The location of the single FASTA file in input

    exclude_proteins : list[str]
        The protein ids to be excluded from the dictionary

    Returns
    -------
    d : dict[str, str]
        All protein ids with their respective annotated sequences
    """
    d = {
        rec.id.split("|")[1] if "|" in rec.id else rec.id: str(rec.seq)
        for rec in SeqIO.parse(open(filename), "fasta")
        if (rec.id.split("|")[1] if "|" in rec.id else rec.id) not in exclude_proteins
    }
    return d


def _read_records(fasta_dict: dict[str, str]) -> tuple:
    """
    Reads annotations from sequence data

    Parameters
    ----------
    fasta_dict : dict[str, str]
        All protein ids with their respective annotated sequences

    Returns
    -------
    data : list[dict[str, Any]]
        Data entries for all protein fragments

    unique_prot_ids : set[str]
        All unique proteins in the dataset

    tot_num_pos : int
        Number of positive annotations in the dataset

    tot_num_neg : int
        Number of negative annotations in the dataset

    """
    data = []
    unique_prot_ids, tot_num_pos, tot_num_neg = set(), 0, 0
    for prot_id in fasta_dict:
        annotated_seq = fasta_dict[prot_id]
        seq_only = annotated_seq.replace("#", "").replace("@", "")
        annots = [-1] * len(seq_only)
        idx = 0
        for i in range(len(annotated_seq)):
            if annotated_seq[i] == "#":
                annots[idx - 1] = 1
            elif annotated_seq[i] == "@":
                annots[idx - 1] = 0
            else:
                idx += 1
        if set(annots) == {-1}:
            # if no annotations or candidates: drop sequence
            continue
        ds, numpos, numneg = _get_annotated_fragments(prot_id, seq_only, annots)
        tot_num_pos += numpos
        tot_num_neg += numneg
        unique_prot_ids.add(prot_id)
        data.extend(ds)
    return data, unique_prot_ids, tot_num_pos, tot_num_neg


# Copied from TAPE repository at https://github.com/songlab-cal/tape/blob/master/tape/datasets.py
def pad_sequences(sequences, constant_value):
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    dtype = sequences[0].dtype
    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


# Given a protein sequence and its annotations, divide it into fragments (default maximum length with (optional)
# extra tokens: 512), with half-overlapping fragments. Annotations will be tied to the fragment where it lies
# closest to the center. In detail:
# (1) The sequence is split up into multiple fragments, which will overlap for half of the sequence
# (2) The last fragment of a sequence will have a bigger overlap, as it will be maximized in length, and span only
#     until the end of the sequence (not beyond)
# (3) Annotations will be allocated to the fragment where they are located the closest to the center
# (*) The maximum sequence length is respected, also taking into account the tokens added at front/back, depending
#     on the encoding strategy
def _get_annotated_fragments(prot_id: str, seq: str, annots: list[int]) -> tuple:
    """
    Given a protein sequence and its annotations, divide it into fragments (default maximum length with (optional)
    extra tokens: 512), with half-overlapping fragments. Annotations will be tied to the fragment where it lies
    closest to the center. In detail:
    (1) The sequence is split up into multiple fragments, which will overlap for half of the sequence
    (2) The last fragment of a sequence will have a bigger overlap, as it will be maximized in length, and span only
        until the end of the sequence (not beyond)
    (3) Annotations will be allocated to the fragment where they are located the closest to the center
    (*) The maximum sequence length is respected, also taking into account the tokens added at front/back, depending
        on the encoding strategy

    Parameters
    ----------
    prot_id : str
        The protein id

    seq : str
        The protein sequence, without annotations

    annots : list[int]
        Annotations for each amino acid in the sequence, either ``1`` (positive), ``2`` (negative), or ``-1`` (not
        annotated)

    Returns
    -------
    ds : list[dict[str, Any]]
        Data entries for all fragments, containing the protein id, the fragment sequence, its offset within the full
        protein, and the labels

    num_pos : int
        Number of positive annotations in all fragments

    num_neg : int
        Number of negative annotations in all fragments
    """
    assert len(seq) == len(annots)
    ds = []
    num_pos, num_neg = 0, 0
    if len(seq) > _PROTEIN_FRAGMENT_LENGTH:
        # example of splitting into fragments:
        # Fragment of length 1337, with max length 512:
        # * num_fragments = 5
        # * from_idx = [0, 256, 512, 768, 825] (last index changed to retain the maximum fragment length)
        # * to_idx = [512, 768, 1024, 1280, 1337]
        num_fragments = int(floor(2 * len(seq) / _PROTEIN_FRAGMENT_LENGTH))
        from_idx = [(_PROTEIN_FRAGMENT_LENGTH) * i // 2 for i in range(num_fragments)]
        to_idx = [
            (_PROTEIN_FRAGMENT_LENGTH) * i // 2 for i in range(2, num_fragments + 2)
        ]
        # last fragment should be as long as allowed, ending at len(seq)
        from_idx[-1], to_idx[-1] = len(seq) - _PROTEIN_FRAGMENT_LENGTH, len(seq)
    else:
        from_idx, to_idx = [0], [len(seq)]

    frag_centers = [(fro + to) // 2 for fro, to in zip(from_idx, to_idx)]
    fragments = [seq[fro:to] for fro, to in zip(from_idx, to_idx)]
    frag_annotations = [[-1 for _ in range(len(fragment))] for fragment in fragments]

    # Link the annotations to the fragments for which they are closest to the center
    for i, c in enumerate(annots):
        if c != -1:
            dist_from_center = [abs(center - i) for center in frag_centers]
            chosen_frag_idx = np.argmin(dist_from_center)
            frag_annotations[chosen_frag_idx][i - from_idx[chosen_frag_idx]] = c

    for frag_seq, annots_for_fragment, from_idx in zip(
        fragments, frag_annotations, from_idx
    ):
        # only include fragments that have annotations
        if set(annots_for_fragment) == {-1}:
            continue
        d = {}
        d["prot_id"] = prot_id
        d["seq_data"] = frag_seq
        d["offset"] = from_idx
        d["labels"] = annots_for_fragment
        num_pos += d["labels"].count(1)
        num_neg += d["labels"].count(0)
        ds.append(d)
    return ds, num_pos, num_neg
