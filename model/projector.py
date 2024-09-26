# import torch
# import torch.nn as nn
# import re
# from config import INSERT_EMBS_TOKEN_ID
# from model.model import Qformer

# IGNORE_INDEX = -100

# class IdentityMap(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, *args, **kwargs):
#         return x

#     @property
#     def config(self):
#         return {"mm_projector_type": 'identity'}


# class SimpleResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.pre_norm = nn.LayerNorm(channels)

#         self.proj = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels)
#         )
#     def forward(self, x):
#         x = self.pre_norm(x)
#         return x + self.proj(x)


# class MultiHeadProjector(nn.Module):

#     def __init__(self, projector_type, encoder_hidden_size, decoder_hidden_size, num_heads, torch_dtype, multihead = True, qformer = True, qformer_dim = 384, qformer_dim_head = 128, qformer_inner_dim = 3584, qformer_query = 10, **kwargs):
#         """
#         Build a table projector based on the given configuration.

#         Args:
#             config (object): mm_projector_type: The type of projector to use. Defaults to 'linear'; hidden_size: ...
#             **kwargs: Additional keyword arguments.

#         Returns:
#             object: The table projector.

#         Raises:
#             ValueError: If the projector type is unknown.
#         """
#         super().__init__()
#         self.multihead = multihead
#         if multihead is False:
#             num_heads = 1
#         self.projector_type = projector_type
#         self.encoder_hidden_size = encoder_hidden_size
#         self.decoder_hidden_size = decoder_hidden_size
#         self.qformer = qformer
#         self.qformer_dim = qformer_dim
#         self.qformer_dim_head = qformer_dim_head
#         self.qformer_inner_dim = qformer_inner_dim
#         self.qformer_query = qformer_query
        
#         self.num_heads = num_heads
#         self.torch_dtype = torch_dtype
        
#         modules = []
#         if qformer:
#             modules.append(Qformer(dim=qformer_dim, dim_head=qformer_dim_head, inner_dim=qformer_inner_dim, query_num=qformer_query))
            
#         mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
#         if mlp_gelu_match:
#             mlp_depth = int(mlp_gelu_match.group(1))
#             modules.append(nn.Linear(encoder_hidden_size, decoder_hidden_size * num_heads, dtype = torch_dtype))
#             for _ in range(1, mlp_depth):
#                 modules.append(nn.GELU())
#                 modules.append(nn.Linear(decoder_hidden_size * num_heads, decoder_hidden_size * num_heads, dtype = torch_dtype))
#         elif projector_type == 'linear':
#             modules.append(nn.Linear(encoder_hidden_size, decoder_hidden_size * num_heads, dtype = torch_dtype))
#         elif projector_type == 'identity':
#             modules.append(IdentityMap())
#             return
#         else: 
#             raise ValueError(f'Unknown projector type: {projector_type}')

#         self.model = nn.Sequential(*modules)

    
#     def forward(self, x):
#         ret = self.model(x)
#         if self.multihead:
#             ret = ret.view(*ret.shape[:-1], self.num_heads, -1)
#         return ret
        

#     def prepare_embeds(
#         self, *, decoder, input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None, table_embeds, learnable_embeds = None
#     ):
#         # Let's just add dummy tensors if they do not exist,
#         # it is a headache to deal with None all the time.
#         # But it is not ideal, and if you have a better idea,
#         # please open an issue / submit a PR, thanks.
#         _labels = labels
#         _position_ids = position_ids
#         _attention_mask = attention_mask
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
#         else:
#             attention_mask = attention_mask.bool()
#         if position_ids is None:
#             position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
#         if labels is None:
#             labels = torch.full_like(input_ids, IGNORE_INDEX)

#         # remove the padding using attention_mask -- FIXME
#         _input_ids = input_ids
#         input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
#         labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


#         new_input_embeds = []
#         new_labels = []
#         for batch_idx, cur_input_ids in enumerate(input_ids):
#             cur_table_embeds = table_embeds[batch_idx].clone()
#             cur_table_embeds = self.model(cur_table_embeds).view(-1, self.decoder_hidden_size) # forward through the projector
#             if learnable_embeds is not None:
#                 cur_table_embeds = torch.cat([cur_table_embeds, learnable_embeds], dim=0)
#             cur_input_embeds = decoder.get_input_embeddings()(cur_input_ids)
#             new_input_embeds.append(torch.cat([cur_table_embeds, cur_input_embeds], dim=0))
#             # new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
#             cur_labels = labels[batch_idx]
#             cur_new_labels = torch.cat((torch.full((cur_table_embeds.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype), cur_labels))
#             new_labels.append(cur_new_labels)

#         # # Truncate sequences to max length as image embeddings can make the sequence longer
#         # tokenizer_model_max_length = getattr(model.config, 'tokenizer_model_max_length', None)
#         # if tokenizer_model_max_length is not None:
#         #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
#         #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

#         # Combine them
#         max_len = max(x.shape[0] for x in new_input_embeds)
#         batch_size = len(new_input_embeds)

#         new_input_embeds_padded = []
#         new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
#         attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
#         position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

#         for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
#             cur_len = cur_new_embed.shape[0]
#             if getattr(decoder.config, 'tokenizer_padding_side', 'right') == "left":
#                 new_input_embeds_padded.append(torch.cat((
#                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
#                     cur_new_embed
#                 ), dim=0))
#                 if cur_len > 0:
#                     new_labels_padded[i, -cur_len:] = cur_new_labels
#                     attention_mask[i, -cur_len:] = True
#                     position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
#             else:
#                 new_input_embeds_padded.append(torch.cat((
#                     cur_new_embed,
#                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
#                 ), dim=0))
#                 if cur_len > 0:
#                     new_labels_padded[i, :cur_len] = cur_new_labels
#                     attention_mask[i, :cur_len] = True
#                     position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

#         new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

#         if _labels is None:
#             new_labels = None
#         else:
#             new_labels = new_labels_padded
#             # new_labels = _labels

#         if _attention_mask is None:
#             pass # keep the newly created attention mask
#         else:
#             attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

#         if _position_ids is None:
#             position_ids = None

#         return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
#     def prepare_insert_embeds(
#         self, *, decoder, input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None, table_embeds, learnable_embeds = None
#     ):
#         assert learnable_embeds == None, "learnable embeddings is not yet supported"
#         # Let's just add dummy tensors if they do not exist,
#         # it is a headache to deal with None all the time.
#         # But it is not ideal, and if you have a better idea,
#         # please open an issue / submit a PR, thanks.
#         _labels = labels
#         _position_ids = position_ids
#         _attention_mask = attention_mask
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
#         else:
#             attention_mask = attention_mask.bool()
#         if position_ids is None:
#             position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
#         if labels is None:
#             labels = torch.full_like(input_ids, IGNORE_INDEX)

#         # remove the padding using attention_mask -- FIXME
#         _input_ids = input_ids
#         input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
#         labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


#         new_input_embeds = []
#         new_labels = []
#         for batch_idx, cur_input_ids in enumerate(input_ids):
#             num_insert_embs = (cur_input_ids == INSERT_EMBS_TOKEN_ID).sum()
#             if num_insert_embs == 0:
#                 raise ValueError("No insert embs token found in the input_ids")
#             cur_table_embeds = table_embeds[batch_idx].clone()
#             cur_table_embeds = self(cur_table_embeds) # forward through the projector
            
#             insert_emb_token_indices = [-1] + torch.where(cur_input_ids == INSERT_EMBS_TOKEN_ID)[0].tolist() + [cur_input_ids.shape[0]]
#             cur_input_ids_noim = []
#             cur_labels = labels[batch_idx]
#             cur_labels_noim = []
#             for i in range(len(insert_emb_token_indices) - 1):
#                 cur_input_ids_noim.append(cur_input_ids[insert_emb_token_indices[i]+1:insert_emb_token_indices[i+1]])
#                 cur_labels_noim.append(cur_labels[insert_emb_token_indices[i]+1:insert_emb_token_indices[i+1]])
#             split_sizes = [x.shape[0] for x in cur_labels_noim]
#             cur_input_embeds = decoder.get_input_embeddings()((torch.cat(cur_input_ids_noim)))
#             cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
#             cur_new_input_embeds = []
#             cur_new_labels = []

#             for i in range(num_insert_embs + 1):
#                 cur_new_input_embeds.append(cur_input_embeds_no_im[i])
#                 cur_new_labels.append(cur_labels_noim[i])
#                 if i < num_insert_embs:
#                     cur_insert_emb_features = cur_table_embeds[i] # num_heads * decode_hidden
#                     if self.multihead:
#                         assert cur_insert_emb_features.shape == (self.num_heads, self.decoder_hidden_size), f"not match: {cur_insert_emb_features.shape}, f{(self.num_heads), self.decoder_hidden_size}"
#                     cur_new_input_embeds.append(cur_insert_emb_features)
#                     cur_new_labels.append(torch.full((cur_insert_emb_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
#             device = next(decoder.parameters()).device
#             cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
#             cur_new_input_embeds = torch.cat(cur_new_input_embeds)
#             cur_new_labels = torch.cat(cur_new_labels)
#             new_input_embeds.append(cur_new_input_embeds)
#             new_labels.append(cur_new_labels)


#         # # Truncate sequences to max length as insert_emb embeddings can make the sequence longer
#         # tokenizer_model_max_length = getattr(model.config, 'tokenizer_model_max_length', None)
#         # if tokenizer_model_max_length is not None:
#         #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
#         #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

#         # Combine them
#         max_len = max(x.shape[0] for x in new_input_embeds)
#         batch_size = len(new_input_embeds)

#         new_input_embeds_padded = []
#         new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
#         attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
#         position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

#         for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
#             cur_len = cur_new_embed.shape[0]
#             if getattr(decoder.config, 'tokenizer_padding_side', 'right') == "left":
#                 new_input_embeds_padded.append(torch.cat((
#                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
#                     cur_new_embed
#                 ), dim=0))
#                 if cur_len > 0:
#                     new_labels_padded[i, -cur_len:] = cur_new_labels
#                     attention_mask[i, -cur_len:] = True
#                     position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
#             else:
#                 new_input_embeds_padded.append(torch.cat((
#                     cur_new_embed,
#                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
#                 ), dim=0))
#                 if cur_len > 0:
#                     new_labels_padded[i, :cur_len] = cur_new_labels
#                     attention_mask[i, :cur_len] = True
#                     position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

#         new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

#         if _labels is None:
#             new_labels = None
#         else:
#             new_labels = new_labels_padded
#             # new_labels = _labels

#         if _attention_mask is None:
#             pass # keep the newly created attention mask
#         else:
#             attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

#         if _position_ids is None:
#             position_ids = None

#         return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels