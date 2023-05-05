import torch
import esm
from typing import Any
from input_tokenizers import TokenAlphabet, tokenizers
from torch import nn
from transformers import T5EncoderModel
from sequence_models.pretrained import load_model_and_alphabet
import ankh


def get_architecture(config: dict[str, Any]) -> torch.nn.Module:
    """
    Returns a neural network architecture based on the given configuration file. By default, this is a convolutional
    neural network, but this can easily be extended by modifying the code in this file.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration for the network architecture

    tokenizer : TokenAlphabet
        Tokenizer to be used in the selected protein language model

    Returns
    -------
    : nn.Module
        The full network architecture to be trained

    """
    return ConvNet(
        representation=config["representation"],
        freeze_representation=config["freeze_representation"],
        dropout=config["dropout"],
        conv_depth=config["conv_depth"],
        conv_width=config["conv_width"],
        conv_channels=config["conv_channels"],
        final_fc_neurons=config["final_fc_neurons"],
        batch_norm=config["batch_norm"],
        max_pool_size=config["max_pool_size"],
        receptive_field=config["receptive_field"],
    )


class ConvNet(torch.nn.Module):
    def __init__(
        self,
        representation: str,
        freeze_representation: bool,
        dropout: float,
        conv_depth: int,
        conv_width: int,
        conv_channels: int,
        final_fc_neurons: int,
        max_pool_size: int,
        batch_norm: bool,
        receptive_field: int,
    ) -> None:
        """
        Convolutional Neural Network architecture, constructed using given hyperparameters

        Parameters
        ----------
        representation : str
            The name of the protein language model to be used

        freeze_representation : bool
            Set to True to disable fine-tuning of the protein language model at training time

        dropout : float
            Dropout chance ``p`` to add to all dropout layers. Will only be added if ``batch_norm == False``

        conv_depth : int
            The number of convolutional blocks in the network

        conv_width : int
            The filter size for the convolutional filters. Must be odd

        conv_channels : int
            The number of output channels for the convolutional layers

        final_fc_neurons : int
            The number of neurons in the fully connected layer at the end of the network

        max_pool_size : int
            The pooling size for the max pooling layers

        batch_norm : bool
            Set to True to enable batch normalization. If True, dropout layers are omitted

        receptive_field : int
            The size of the receptive field in input, centered around the candidate P-site. Must be odd

        Attributes
        ----------
        representation : str
            The name of the protein language model used

        receptive_field: int
            The size of the receptive field in input, centered around the candidate P-site. Must be odd

        tokenizer : TokenAlphabet
            The tokenizer to be used for the selected protein language model

        encoding : torch.nn.Module
            The encoding, either via a one-hot encoding or a protein language model

        conv_network : torch.nn.Module
            The convolutional blocks in the network

        output_block : torch.nn.Module
            The output block at the end of the network, including fully connected layers
        """
        super().__init__()
        assert receptive_field % 2 == 1
        assert conv_width % 2 == 1
        self.representation = representation
        self.receptive_field = receptive_field
        self.tokenizer = tokenizers[representation]()

        if representation == "onehot":
            self.encoding = OneHotEncoding(self.tokenizer)
        else:
            self.encoding = LanguageModel(
                representation=representation,
                tokenizer=self.tokenizer,
                freeze_representation=freeze_representation,
            )

        self.conv_network = ConvolutionBlock(
            num_input_channels=self.encoding.get_num_channels(),
            input_width=receptive_field,
            conv_channels=conv_channels,
            conv_width=conv_width,
            conv_depth=conv_depth,
            dropout=dropout,
            batch_norm=batch_norm,
            max_pool_size=max_pool_size,
        )

        # the output width of the conv network will be the receptive field, reduced by each pooling layer
        dense_input_width = receptive_field
        for _ in range(conv_depth):
            dense_input_width = dense_input_width // max_pool_size

        self.output_block = DenseBlock(
            input_width=dense_input_width,
            num_input_channels=self.conv_network.get_num_channels(),
            num_neurons=final_fc_neurons,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    # When calling this forward function, the targeted positions (= positions for which prediction are to be made) need
    # to be specified. Using this information, an output will be generated for each candidate site, using the context
    # within the receptive field surrounding each position.
    # Example: an input of size (32, 512) is given, with on average exactly 3 candidate phospho-sites. The output
    # logits will have size (96, 2), with a negative/positive logit for each candidate site.
    # The embedding layer is run only once per input sequence, as this can be computationally expensive when working
    # with big transformer models. After the embedding, the convolutional and max pooling layers are processed for each
    # candidate site separately.
    def forward(self, x, x_mask_per_seq, x_mask_per_seq_no_extra, targeted_positions) -> torch.tensor:
        """
        Forward pass of the full network. When calling this forward function, the targeted positions (= positions for which prediction are to be made) need
         to be specified. Using this information, an output will be generated for each candidate site, using the context
         within the receptive field surrounding each position.
         Example: an input of size (32, 512) is given, with on average exactly 3 candidate phospho-sites. The output
         logits will have size (96, 2), with a negative/positive logit for each candidate site.
         The embedding layer is run only once per input sequence, as this can be computationally expensive when working
         with big transformer models. After the embedding, the convolutional and max pooling layers are processed for each
         candidate site separately.

        Parameters
        ----------
        x : torch.tensor
            The input tokens for a given batch

        x_mask : torch.tensor
            The mask for the input tokens, including extra tokens added in input

        x_mask_no_extra : torch.tensor
            The mask for the input tokens, excluding extra tokens added in input

        targeted_positions : torch.tensor
            An all-zero tensor with 1's at the positions for which predictions are to be made
        """
        x = self.encoding(
            x, x_mask_per_seq, x_mask_per_seq_no_extra
        )  # out: (batch_size, batch_seqlen, embedding_size)
        padded_x = torch.nn.functional.pad(
            input=x,
            pad=[0, 0, self.receptive_field // 2, self.receptive_field // 2, 0, 0],
            mode="constant",
            value=0,
        )
        bs, ln, es = x.shape
        # For each candidate site, repeat the embedding calculated for the corresponding fragment:
        # (a) create lists with indices for each candidate site, within the fragment!
        idx_in_seq = (
            torch.arange(ln, device=x.device).unsqueeze(0).repeat(bs, 1)
        )  # (batch_size, batch_seqlen)
        # (b) select the indices that are targeted
        idx_in_seq = idx_in_seq[targeted_positions == 1]  # (num_targets_in_batch,)
        # (c) create lists with indices for each candidate site, within the batch!
        idx_in_batch = (
            torch.arange(bs, device=x.device).unsqueeze(1).repeat(1, ln)
        )  # (batch_size, batch_seqlen)
        # (d) select the batch indices of targeted sites
        idx_in_batch = idx_in_batch[targeted_positions == 1]  # (num_targets_in_batch,)

        selected_x = padded_x[idx_in_batch]
        # For each candidate site, gather the indices that are within the receptive field surrounding the site. This is
        # done by adding a list of values between 0 and receptive_field, to the index of each site. Then, the gather
        # function is used to keep just that part of the sequence
        receptive_field_ranges = torch.arange(
            self.receptive_field, device=x.device
        ).unsqueeze(
            0
        )  # (1, receptive_field)
        receptive_field_indices = (
            idx_in_seq.unsqueeze(1) + receptive_field_ranges
        )  # (num_targets_in_batch, receptive_field)
        x_conv_block_inputs = selected_x.gather(
            dim=1, index=receptive_field_indices.unsqueeze(-1).repeat(1, 1, es)
        )

        x = self.conv_network(
            x_conv_block_inputs
        )  # out: (num_targets_in_batch, receptive_field // (conv_depth*max_pool_size), num_conv_filters)
        x = self.output_block(x)  # out: (num_targets_in_batch, 1)
        return x.squeeze(-1)

    def get_tokenizer(self):
        return self.tokenizer


class OneHotEncoding(torch.nn.Module):
    def __init__(self, tokenizer: TokenAlphabet) -> None:
        """
        One-hot encoding, using a PyTorch embedding layer to go from tokens to representation

        Parameters
        ----------
        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation

        Attributes
        ----------
        vocab_size : int
            The number of possible tokens in input, and thus the size of the one-hot encoded vectors

        embedding_model : torch.nn.Module
            The one-hot encoding
        """
        super(OneHotEncoding, self).__init__()
        self.vocab_size = len(tokenizer)
        onehot_encoding = torch.zeros(self.vocab_size, self.vocab_size - 1)

        for i in range(self.vocab_size - 1):  # -1 because padding gets all-zeros
            onehot_encoding[i][i] = 1

        self.embedding_model = torch.nn.Embedding.from_pretrained(
            onehot_encoding, freeze=True
        )

    def forward(self, x, *args):
        x = self.embedding_model(x)
        return x

    def get_num_channels(self):
        return self.vocab_size - 1


# If the encoding is set to be one of the protein language models, that model is (down)loaded here, and some extra
# parameters are initialized
class LanguageModel(torch.nn.Module):
    def __init__(self, representation, tokenizer, freeze_representation=True):
        """
        A protein language model representation. Language models are downloaded and initialized in this class.

        Parameters
        ----------
        representation : str
            The name of the protein language model used

        tokenizer : TokenAlphabet
            The tokenizer to be used for the selected protein language model

        freeze_representation: bool
            Set to True to disable fine-tuning of the protein language model at training time

        Attributes
        ----------
        representation : str
            The name of the protein language model used

        tokenizer : TokenAlphabet
            The tokenizer to be used for the selected protein language model

        embedding_model : torch.nn.Module
            The language model to call with to compute embeddings

        channels : int
            The number of output channels after applying the protein language model

        esm_last_layer_idx : int
            In case of ESM models, the index of the output layer from which to extract the representation

        """
        super(LanguageModel, self).__init__()
        self.representation = representation
        self.tokenizer = tokenizer
        if representation == "ESM1b":
            self.embedding_model, _ = esm.pretrained.esm1b_t33_650M_UR50S()
            self.channels = 1280
            self.esm_last_layer_idx = 33
        elif representation == "ESM1_small":
            self.embedding_model, _ = esm.pretrained.esm1_t6_43M_UR50S()
            self.channels = 768
            self.esm_last_layer_idx = 6
        elif representation == 'ESM2_150M':
            self.embedding_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
            self.channels = 640
            self.esm_last_layer_idx = 30
        elif representation == 'ESM2_650M':
            self.embedding_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
            self.channels = 1280
            self.esm_last_layer_idx = 33
        elif representation == 'ESM2_3B':
            self.embedding_model, _ = esm.pretrained.esm2_t36_3B_UR50D()
            self.channels = 2560
            self.esm_last_layer_idx = 36
        elif representation == 'ESM2_15B':
            self.embedding_model, _ = esm.pretrained.esm2_t48_15B_UR50D()
            self.channels = 5120
            self.esm_last_layer_idx = 48
        elif representation == 'CARP_640M':
            self.embedding_model, _ = load_model_and_alphabet('carp_640M')
            self.channels = 1280
        elif representation == "ProtTransT5_XL_UniRef50":
            self.embedding_model = T5EncoderModel.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50"
            )
            self.channels = 1024
        elif representation == 'Ankh_base':
            self.embedding_model, _ = ankh.load_base_model()
            self.channels = 768
        elif representation == 'Ankh_large':
            self.embedding_model, _ = ankh.load_large_model()
            self.channels = 1536
        else:
            raise NotImplementedError(
                f'"{representation}" not supported as embedding type'
            )

        # if the representation needs to be 'frozen', set encoding model parameters untrainable
        if freeze_representation:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def forward(self, x, x_mask_per_seq=None, x_mask_per_seq_no_extra=None) -> torch.tensor:
        """
        Forward pass of the protein language model. The ``x_mask_no_extra`` takes care of zero'ing out padded tokens,
        and removes appended/prepended extra, non-amino acid tokens

        Parameters
        ----------
        x : torch.tensor
            The input tokens for a given batch

        x_mask : torch.tensor
            The mask for the input tokens, including extra tokens added in input

        x_mask_no_extra : torch.tensor
            The mask for the input tokens, excluding extra tokens added in input

        Returns
        -------
        rep : torch.tensor
            The protein representation, excluding extra tokens, for a given tokenized input
        """
        if "ESM" in self.representation:
            results = self.embedding_model(
                x, repr_layers=[self.esm_last_layer_idx], return_contacts=False
            )
            rep = results["representations"][self.esm_last_layer_idx]
            rep = rep * x_mask_per_seq_no_extra.unsqueeze(-1)
            rep = rep[:, self.tokenizer.get_num_tokens_added_front() :]
            if self.tokenizer.get_num_tokens_added_back():
                rep = rep[:, : -self.tokenizer.get_num_tokens_added_back()]
            return rep
        elif self.representation == "ProtTransT5_XL_UniRef50":
            x = self.embedding_model(input_ids=x, attention_mask=x_mask_per_seq)
            rep = x.last_hidden_state * x_mask_per_seq_no_extra.unsqueeze(-1)
            rep = rep[:, self.tokenizer.get_num_tokens_added_front() :]
            if self.tokenizer.get_num_tokens_added_back():
                rep = rep[:, : -self.tokenizer.get_num_tokens_added_back()]
            return rep
        else:
            raise NotImplementedError

    def get_num_channels(self) -> int:
        return self.channels


class ConvolutionBlock(torch.nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        input_width: int,
        conv_channels: int,
        conv_width: int,
        conv_depth: int,
        dropout: float,
        batch_norm: bool,
        max_pool_size: int,
    ) -> None:
        """
        The convolutional neural network block, consisting by a collection of convolutional, max pooling and dropout or
        batch normalization layers

        Parameters
        ----------
        num_input_channels : int
            The number of output channels of the used protein representation, given in input to this ConvolutionBlock

        input_width : int
            The width of the input to this ConvolutionBlock. This depends on the receptive field in the configuration

        conv_channels: int
            The number of output channels for the convolutional layers

        conv_width: int
            The filter size for the convolutional filters. Must be odd

        conv_depth: int
            The number of convolutional blocks in the network

        dropout: float
            Dropout chance ``p`` to add to all dropout layers. Will only be added if ``batch_norm == False``

        batch_norm: bool
            Set to True to enable batch normalization. If True, dropout layers are omitted

        Attributes
        ----------
        l : nn.ModuleList[nn.Module]
            All layers in the convolutional block

        num_channels : int
            The number of output channels at each step in the convolutional blocks; will be modified by convolutional
            layers

        width : int
            The width at each step in the convolutional blocks; will be modified by max pooling operations
        """
        super(ConvolutionBlock, self).__init__()
        self.l = nn.ModuleList()
        self.num_channels = num_input_channels
        self.width = input_width
        for _ in range(conv_depth):
            self.l.append(
                nn.Conv1d(
                    self.num_channels,
                    conv_channels,
                    conv_width,
                    padding=conv_width // 2,
                )
            )
            self.l.append(nn.ReLU())
            # if batch_norm is True, it overrides dropout
            if batch_norm:
                self.l.append(nn.BatchNorm1d(conv_channels))
            else:
                self.l.append(nn.Dropout(p=dropout))
            if max_pool_size > 1:
                self.l.append(nn.MaxPool1d(max_pool_size))
                self.width = self.width // max_pool_size
            self.num_channels = conv_channels

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.permute(0, 2, 1)  # put channels at dim 1
        for func in self.l:
            x = func(x)
        x = x.permute(0, 2, 1)  # put channels at dim 2
        return x

    def get_num_channels(self) -> int:
        return self.num_channels

    def get_output_width(self) -> int:
        return self.width


class DenseBlock(torch.nn.Module):
    def __init__(
        self,
        input_width: int,
        num_input_channels: int,
        num_neurons: int,
        dropout: float,
        batch_norm: bool,
    ) -> None:
        """
        A Dense block flattens the input, adds a dense layer and a dropout/batch norm layer (if num_neurons != 0),
        followed by an output layer with one neuron. The results are logits.

        Parameters
        ----------
        input_width : int
            The width of x after applying the convolutional block

        num_input_channels : int
            The number of output channels after applying the convolutional block

        num_neurons : int
            The number of neurons in the fully connected layer at the end of the network

        dropout : float
            Dropout chance ``p`` to add to all dropout layers. Will only be added if ``batch_norm == False``

        batch_norm : bool
            Set to True to enable batch normalization. If True, dropout layers are omitted

        Attributes
        ----------
        in_features : int
            The size of the input to the DenseBlock after flattening

        l : nn.ModuleList[nn.Module]
            All layers in the dense block
        """
        super(DenseBlock, self).__init__()
        self.in_features = input_width * num_input_channels
        last_num_channels = self.in_features
        self.l = nn.ModuleList()
        if (
            num_neurons
        ):  # if 0, skip the fully-connected layer, and go straight to the output layer
            self.l.append(nn.Linear(last_num_channels, num_neurons))
            last_num_channels = num_neurons
            self.l.append(nn.ReLU())
            if batch_norm:
                self.l.append(nn.BatchNorm1d(num_neurons))
            else:
                self.l.append(nn.Dropout(p=dropout))
        self.l.append(nn.Linear(last_num_channels, 1))

    def forward(self, x):
        x = x.reshape((x.shape[0], self.in_features))
        for func in self.l:
            x = func(x)
        return x
