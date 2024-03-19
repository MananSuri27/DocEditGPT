import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from transformers import Pix2StructTextModel, Pix2StructConfig, Pix2StructVisionModel, MaskFormerConfig, DetrConfig
from docmasktransformer import DocMaskTransformer
import torch.nn.functional as F

def dice_loss_fun(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    probs = inputs.flatten(1)
    labels = labels.flatten(1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.mean().sum() / num_masks
    return loss


def sigmoid_focal_loss_fun(
    inputs: Tensor, labels: Tensor, num_masks: int, alpha: float = 0.25, gamma: float = 2
) -> Tensor:
    r"""
    Focal loss proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) originally used in
    RetinaNet. The loss is computed as follows:

    $$ \mathcal{L}_{\text{focal loss} = -(1 - p_t)^{\gamma}\log{(p_t)} $$

    where \\(CE(p_t) = -\log{(p_t)}}\\), CE is the standard Cross Entropy Loss

    Please refer to equation (1,2,3) of the paper for a better understanding.

    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.
        alpha (float, *optional*, defaults to 0.25):
            Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float, *optional*, defaults to 2.0):
            Exponent of the modulating factor \\(1 - p_t\\) to balance easy vs hard examples.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    inputs = inputs.flatten(1)
    labels = labels.flatten(1)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probs = inputs
    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(1).sum() / num_masks
    return loss

def is_torch_fx_proxy(x):
    try:
        import torch.fx
        return isinstance(x, torch.fx.Proxy)
    except ImportError:
        return False

class Doc2Command(nn.Module):
    """
        Multi-task model to generate command and identify region of interest from user request and input document. 

    """
    config_class = Pix2StructConfig
    main_input_name = "flattened_patches"
    _tied_weights_keys = ["decoder.lm_head.weight"]

    def __init__(self, config):
        super().__init__()

        self.config = config["p2s"]

        self.encoder = Pix2StructVisionModel(config["p2s"].vision_config)
        self.decoder = Pix2StructTextModel(config["p2s"].text_config)
        self.masktransformer = DocMaskTransformer(**config["masktransformer_config"])
        self.mask_loss = nn.BCELoss()

        self.initialse_weights()


        # # Initialize weights and apply final processing
        # self.post_init()
    def initialse_weights(self):
        weights = torch.load("pytorch_model.bin")
        self.encoder.load_state_dict({".".join(r.split(".")[1:]):weights[r] for r in weights if r.startswith("encoder")})
        self.decoder.load_state_dict({".".join(r.split(".")[1:]):weights[r] for r in weights if r.startswith("decoder")})

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)

        # update vocab size
        self.config.text_config.vocab_size = new_num_tokens

        return model_embeds

    def get_decoder(self):
        return self.decoder

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ground_truth_mask = None,
        shape =None
    ):

        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


        hidden_states = encoder_outputs[0]


       # Output of this will be segmentation map [batch_size, num_labels, height, width], we can argmax over second dimension to find output
        masks = self.masktransformer(encoder_outputs["last_hidden_state"][:,:int(attention_mask.sum().item()),:], flattened_patches[:,:,0][0].max())
        masks = F.interpolate(masks, size=(shape[0], shape[1]), mode="bilinear")



        masks = torch.sigmoid(masks)


        dice_loss = dice_loss_fun(masks, ground_truth_mask, 3)
        focal_loss = sigmoid_focal_loss_fun(masks, ground_truth_mask, 3)

        mask_loss = dice_loss+focal_loss

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = (
                decoder_attention_mask
                if decoder_attention_mask is not None
                else decoder_input_ids.ne(self.config.pad_token_id).float()
            )
            # Always attend to the first token
            decoder_attention_mask[:, 0] = 1

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return {
                "loss": self.config.lambda1*decoder_outputs.loss+ self.config.lambda2*mask_loss,
                "decoder_loss":decoder_outputs.loss,
                "mask_loss": mask_loss,
                "mask": masks,
                "logits": decoder_outputs.logits,
                "past_key_values": decoder_outputs.past_key_values,
                "decoder_hidden_states": decoder_outputs.hidden_states,
                "decoder_attentions": decoder_outputs.attentions,
                "cross_attentions": decoder_outputs.cross_attentions,
                "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
                "encoder_hidden_states": encoder_outputs.hidden_states,
                "encoder_attentions": encoder_outputs.attentions,
            }

    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id."
                "See Pix2Struct docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids