from typing import Optional, Tuple, Union, List
import time

import torch
import warnings
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torch.distributed as dist
from transformers.models.gpt2.modeling_gpt2 import  GPT2LMHeadModel, CausalLMOutputWithCrossAttentions, GPT2Model, BaseModelOutputWithPastAndCrossAttentions,GPT2PreTrainedModel, ModelOutput, dataclass, GPT2Attention, GPT2Block
from transformers.utils import (
    logging, 
)
from transformers.generation_utils import LogitsProcessorList, StoppingCriteriaList, GreedySearchOutput, validate_stopping_criteria, GreedySearchEncoderDecoderOutput

logger = logging.get_logger(__name__)

class GPT2AttentionWithPropagation(GPT2Attention):
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        just_propagate: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)


        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if just_propagate:
            return (key, value)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2BlockWithPropagation(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        self.attn = GPT2AttentionWithPropagation(config, layer_idx=layer_idx)
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        just_propagate: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            just_propagate=just_propagate)
        if just_propagate:
            return attn_outputs
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)



@dataclass
class GreedySearchDecoderOnlyOutputWithExit(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    exit_layers: Optional[List] = None
    is_stopped: Optional[bool] = None

class CausalLMOutputWithCrossAttentionsAndExit(CausalLMOutputWithCrossAttentions):
    def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None, cross_attentions=None, exit_layers=None, is_stopped=None):
        super().__init__(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states, attentions=attentions, cross_attentions=cross_attentions)
        self.exit_layers = exit_layers
        self.is_stopped = is_stopped

class GPT2ModelWithExit(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2BlockWithPropagation(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.current_exit_layer = config.n_layer
        self.is_stopped = False
        self.post_init()

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_layer: Optional[list] = None,
        stop_threshold: Optional[float]=0.0,
        exit_threshold: Optional[float]=0.0,
        return_dict: Optional[bool] = None,
        is_first_token: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        

        if output_layer is not None and batch_size != 1:
            raise ValueError("output_layer is only supported for batch_size = 1")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        self.current_exit_layer = self.config.n_layer
        self.is_stopped = False
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if not self.training and output_layer is not None and not is_first_token:
                exit_logits = output_layer(hidden_states[0,-1,:])
                confident = torch.softmax(exit_logits, dim=-1)
                max_cls = torch.argmax(confident, dim=-1)
                # label 0 is stop
                # label 1 is exit
                # label 2 is continue
                self.current_exit_layer = i + 1
                if confident[0] > stop_threshold and max_cls == 0:
                    # end the inference
                    self.is_stopped = True
                    break
                elif confident[1] > exit_threshold and max_cls == 1:
                    # early exit
                    if use_cache is True:
                        presents = presents + (outputs[1],)
                        # generate key/value of future layer based on current hidden states
                        for j in range(i + 1, self.config.n_layer):
                            kv = self.h[j](hidden_states, past_key_values[j], just_propagate=True)
                            
                            presents = presents + (kv,)
                        
                    if output_attentions:
                        raise NotImplementedError
                        all_self_attentions = all_self_attentions + (*[outputs[2 if use_cache else 1] for _ in range(self.config.n_layer-i)],)
                        if self.config.add_cross_attention:
                            all_cross_attentions = all_cross_attentions +  (*[outputs[3 if use_cache else 2] for _ in range(self.config.n_layer-i)],)
                    break
                
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        

class GPT2MultiLMHeadModelWithExit(GPT2PreTrainedModel):
    def __init__(self, config, mode='eval'):
        super().__init__(config)
        self.transformer = GPT2ModelWithExit(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.inter_lm_head = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.n_layer - 1)])
        # self.inter_ln_f = nn.ModuleList([nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon) for _ in range(config.n_layer - 1)])
        self.exit_classifier = nn.Linear(config.n_embd, 3, bias=False) 
        self.loss_fct = CrossEntropyLoss()
        
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.mode = mode
        if mode == 'train_classifer':
            self.lm_head.eval()
            for param in self.lm_head.parameters():
                param.requires_grad = False
            for inter in self.inter_lm_head:
                inter.eval()
                for param in inter.parameters():
                    param.requires_grad = False 
            # for inter in self.inter_ln_f:
            #     inter.eval()
            #     for param in inter.parameters():
            #         param.requires_grad = False 
        elif mode == 'train_lm':
            self.exit_classifier.eval()
            for param in self.exit_classifier.parameters():
                param.requires_grad = False
         # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_all_layers: Optional[bool] = None,
        return_exit_logits: Optional[bool] = None,
        use_exit: Optional[bool] = None,
        stop_threshold: Optional[float]=100.0,
        exit_threshold: Optional[float]=100.0,
        is_first_token: Optional[bool]=True
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_layer=self.exit_classifier if use_exit else None,
            return_dict=return_dict,
            stop_threshold=stop_threshold,
            exit_threshold=exit_threshold,
            is_first_token=is_first_token
        )
        hidden_states = transformer_outputs[0]
        all_hidden_states = transformer_outputs[2]
        all_hidden_states = all_hidden_states[1:]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        lm_logits = None
        if labels is not None and self.mode == 'train_classifier':
            total_weight = 0
            for i in range(self.config.n_layer):
                current_label = labels[..., i, :].contiguous()
                exit_logits = self.exit_classifier(all_hidden_states[i])
                if loss is None:
                    loss = self.loss_fct(exit_logits.view(-1, exit_logits.size(-1)), current_label.view(-1))
                else:
                    loss += self.loss_fct(exit_logits.view(-1, exit_logits.size(-1)), current_label.view(-1)) * (i+1)
                total_weight += i + 1
            loss /= total_weight
        elif labels is not None and self.mode == 'train_lm':
            shift_labels = labels[..., 1:].contiguous()
            for i in range(len(self.inter_lm_head)):
                # normalized = self.inter_ln_f[i](all_hidden_states[i])
                # lm_logits = self.inter_lm_head[i](normalized)
                lm_logits = self.inter_lm_head[i](all_hidden_states[i])
                shift_inter_lm_logits = lm_logits[..., :-1, :].contiguous()
                if loss is None:
                    loss = self.loss_fct(shift_inter_lm_logits.view(-1, shift_inter_lm_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss += self.loss_fct(shift_inter_lm_logits.view(-1, shift_inter_lm_logits.size(-1)), shift_labels.view(-1))
        elif use_exit:
            if self.transformer.is_stopped:
                x,y,_ = hidden_states.size()
                lm_logits = torch.ones((x,y,self.config.vocab_size)).to(input_ids.device)
                lm_logits[:,-1,self.config.eos_token_id] = 100
            elif self.transformer.current_exit_layer == self.config.n_layer:
                lm_logits = self.lm_head(self.transformer.ln_f(hidden_states))
            else:
                # normalized = self.inter_ln_f[self.transformer.current_exit_layer -1](hidden_states)
                # lm_logits = self.inter_lm_head[self.transformer.current_exit_layer -1](normalized)
                lm_logits = self.inter_lm_head[self.transformer.current_exit_layer -1](hidden_states)
        else:
            lm_logits = self.lm_head(self.transformer.ln_f(hidden_states))


        if return_all_layers:
            assert labels is not None
            for i in range(len(self.inter_lm_head)):
                # normalized = self.inter_ln_f[i](all_hidden_states[i])
                # inter_lm_logits = self.inter_lm_head[i](normalized)
                inter_lm_logits = self.inter_lm_head[i](all_hidden_states[i])
                if i == 0:
                    all_inter_lm_logits = [inter_lm_logits]
                else:
                    all_inter_lm_logits.append(inter_lm_logits)
            all_inter_lm_logits.append(lm_logits)
            lm_logits = all_inter_lm_logits
        
        if return_exit_logits:
            if return_all_layers:
                raise ValueError("return_all_layers and return_exit_logits cannot be both True")
            for i in range(self.config.n_layer):
                exit_logits = self.exit_classifier(all_hidden_states[i])
                if i == 0:
                    all_exit_logits = [exit_logits]
                else:
                    all_exit_logits.append(exit_logits)
            lm_logits = all_exit_logits
        
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentionsAndExit(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            exit_layers=self.transformer.current_exit_layer,
            is_stopped=self.transformer.is_stopped
        )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        exit_layers = []
        is_stopped = False

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        is_first_token = True
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                is_first_token=is_first_token,
                output_attentions=output_attentions,
                use_exit=model_kwargs['use_exit'],
                output_hidden_states=output_hidden_states,
                stop_threshold=model_kwargs['stop_threshold'],
                exit_threshold=model_kwargs['exit_threshold'],
            )
            is_first_token = False
            is_stopped = outputs.is_stopped
            exit_layers.append(outputs.exit_layers)
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutputWithExit(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    exit_layers=exit_layers,
                    is_stopped=is_stopped
                )
        else:
            return input_ids
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }