import torch
results_dict = torch.load('/home/ch7858/vpt/output_val/vtab-caltech101_P32_VK5_SHARED_1_INIT_1_ACC_0/sup_vitb16_224/lr12.5_wd0.01/run1/last_model.pth', "cpu")
# # val_result = results_dict[f"epoch_{100}"]["classification"][t_name]["top1"]
# # val_result = float(val_result)
# exit()

def determine_mask_sequence(n_pieces_token=16, n_soft_tokens=20):
    # soft_token_mask_number = args.soft_token_mask_number
    soft_token_mask_number = None
    if soft_token_mask_number is None:
        soft_token_mask_number = []
        for prune_percent in "seq 5 5 100":
            total_soft_tokens = n_soft_tokens
            n_to_mask = int(total_soft_tokens * prune_percent / 100)
            if args.min_num_soft_tokens > 0:
                if n_to_mask > total_soft_tokens - args.min_num_soft_tokens:
                    n_to_mask = total_soft_tokens - args.min_num_soft_tokens
                    soft_token_mask_number.append(n_to_mask)
                    break
            soft_token_mask_number.append(n_to_mask)
    soft_token_mask_number = sorted(soft_token_mask_number)
    soft_token_mask_sequence = soft_token_mask_number[:]
    for idx in range(1, len(soft_token_mask_number)):
        soft_token_mask_sequence[idx] = soft_token_mask_number[idx] - soft_token_mask_number[idx-1]
    assert soft_token_mask_number[-1] == sum(soft_token_mask_sequence)
    
    token_piece_mask_number = args.token_piece_mask_number
    if token_piece_mask_number is None:
        token_piece_mask_number = []
        for prune_percent in args.mask_token_pieces_percent:
            total_soft_tokens_pieces = n_pieces_token
            n_to_mask = int(total_soft_tokens_pieces * prune_percent / 100)
            if args.min_num_soft_tokens_pieces > 0:
                if n_to_mask > total_soft_tokens_pieces - args.min_num_soft_tokens_pieces:
                    n_to_mask = total_soft_tokens_pieces - args.min_num_soft_tokens_pieces
                    token_piece_mask_number.append(n_to_mask)
                    break
            token_piece_mask_number.append(n_to_mask)
    token_piece_mask_number = sorted(token_piece_mask_number)
    token_piece_mask_sequence = token_piece_mask_number[:]
    for idx in range(1, len(token_piece_mask_number)):
        token_piece_mask_sequence[idx] = token_piece_mask_number[idx] - token_piece_mask_number[idx-1]
    
    assert token_piece_mask_number[-1] == sum(token_piece_mask_sequence)
    return soft_token_mask_sequence, token_piece_mask_sequence

# determine_mask_sequence()


1. should pass here!
2. go through vit_models.py -- P_VK(Prompt with value and key)
3. Go through P_VK_cfg --- build_mocov3_model.py
##### cls_token
False
##### pos_embed
False
##### prompt_embeddings
##### deep_prompt_embeddings
##### prompt_soft_tokens_mask_cls_token
##### prompt_soft_tokens_pieces_mask_cls_token
##### patch_embed.proj.weight
False
##### patch_embed.proj.bias
False
##### blocks.0.norm1.weight
False
##### blocks.0.norm1.bias
False
##### blocks.0.attn.qkv.weight
False
##### blocks.0.attn.qkv.bias
False
##### blocks.0.attn.proj.weight
False
##### blocks.0.attn.proj.bias
False
##### blocks.0.norm2.weight
False
##### blocks.0.norm2.bias
False
##### blocks.0.mlp.fc1.weight
False
##### blocks.0.mlp.fc1.bias
False
##### blocks.0.mlp.fc2.weight
False
##### blocks.0.mlp.fc2.bias
False
##### blocks.1.norm1.weight
False
##### blocks.1.norm1.bias
False
##### blocks.1.attn.qkv.weight
False
##### blocks.1.attn.qkv.bias
False
##### blocks.1.attn.proj.weight
False
##### blocks.1.attn.proj.bias
False
##### blocks.1.norm2.weight
False
##### blocks.1.norm2.bias
False
##### blocks.1.mlp.fc1.weight
False
##### blocks.1.mlp.fc1.bias
False
##### blocks.1.mlp.fc2.weight
False
##### blocks.1.mlp.fc2.bias
False
##### blocks.2.norm1.weight
False
##### blocks.2.norm1.bias
False
##### blocks.2.attn.qkv.weight
False
##### blocks.2.attn.qkv.bias
False
##### blocks.2.attn.proj.weight
False
##### blocks.2.attn.proj.bias
False
##### blocks.2.norm2.weight
False
##### blocks.2.norm2.bias
False
##### blocks.2.mlp.fc1.weight
False
##### blocks.2.mlp.fc1.bias
False
##### blocks.2.mlp.fc2.weight
False
##### blocks.2.mlp.fc2.bias
False
##### blocks.3.norm1.weight
False
##### blocks.3.norm1.bias
False
##### blocks.3.attn.qkv.weight
False
##### blocks.3.attn.qkv.bias
False
##### blocks.3.attn.proj.weight
False
##### blocks.3.attn.proj.bias
False
##### blocks.3.norm2.weight
False
##### blocks.3.norm2.bias
False
##### blocks.3.mlp.fc1.weight
False
##### blocks.3.mlp.fc1.bias
False
##### blocks.3.mlp.fc2.weight
False
##### blocks.3.mlp.fc2.bias
False
##### blocks.4.norm1.weight
False
##### blocks.4.norm1.bias
False
##### blocks.4.attn.qkv.weight
False
##### blocks.4.attn.qkv.bias
False
##### blocks.4.attn.proj.weight
False
##### blocks.4.attn.proj.bias
False
##### blocks.4.norm2.weight
False
##### blocks.4.norm2.bias
False
##### blocks.4.mlp.fc1.weight
False
##### blocks.4.mlp.fc1.bias
False
##### blocks.4.mlp.fc2.weight
False
##### blocks.4.mlp.fc2.bias
False
##### blocks.5.norm1.weight
False
##### blocks.5.norm1.bias
False
##### blocks.5.attn.qkv.weight
False
##### blocks.5.attn.qkv.bias
False
##### blocks.5.attn.proj.weight
False
##### blocks.5.attn.proj.bias
False
##### blocks.5.norm2.weight
False
##### blocks.5.norm2.bias
False
##### blocks.5.mlp.fc1.weight
False
##### blocks.5.mlp.fc1.bias
False
##### blocks.5.mlp.fc2.weight
False
##### blocks.5.mlp.fc2.bias
False
##### blocks.6.norm1.weight
False
##### blocks.6.norm1.bias
False
##### blocks.6.attn.qkv.weight
False
##### blocks.6.attn.qkv.bias
False
##### blocks.6.attn.proj.weight
False
##### blocks.6.attn.proj.bias
False
##### blocks.6.norm2.weight
False
##### blocks.6.norm2.bias
False
##### blocks.6.mlp.fc1.weight
False
##### blocks.6.mlp.fc1.bias
False
##### blocks.6.mlp.fc2.weight
False
##### blocks.6.mlp.fc2.bias
False
##### blocks.7.norm1.weight
False
##### blocks.7.norm1.bias
False
##### blocks.7.attn.qkv.weight
False
##### blocks.7.attn.qkv.bias
False
##### blocks.7.attn.proj.weight
False
##### blocks.7.attn.proj.bias
False
##### blocks.7.norm2.weight
False
##### blocks.7.norm2.bias
False
##### blocks.7.mlp.fc1.weight
False
##### blocks.7.mlp.fc1.bias
False
##### blocks.7.mlp.fc2.weight
False
##### blocks.7.mlp.fc2.bias
False
##### blocks.8.norm1.weight
False
##### blocks.8.norm1.bias
False
##### blocks.8.attn.qkv.weight
False
##### blocks.8.attn.qkv.bias
False
##### blocks.8.attn.proj.weight
False
##### blocks.8.attn.proj.bias
False
##### blocks.8.norm2.weight
False
##### blocks.8.norm2.bias
False
##### blocks.8.mlp.fc1.weight
False
##### blocks.8.mlp.fc1.bias
False
##### blocks.8.mlp.fc2.weight
False
##### blocks.8.mlp.fc2.bias
False
##### blocks.9.norm1.weight
False
##### blocks.9.norm1.bias
False
##### blocks.9.attn.qkv.weight
False
##### blocks.9.attn.qkv.bias
False
##### blocks.9.attn.proj.weight
False
##### blocks.9.attn.proj.bias
False
##### blocks.9.norm2.weight
False
##### blocks.9.norm2.bias
False
##### blocks.9.mlp.fc1.weight
False
##### blocks.9.mlp.fc1.bias
False
##### blocks.9.mlp.fc2.weight
False
##### blocks.9.mlp.fc2.bias
False
##### blocks.10.norm1.weight
False
##### blocks.10.norm1.bias
False
##### blocks.10.attn.qkv.weight
False
##### blocks.10.attn.qkv.bias
False
##### blocks.10.attn.proj.weight
False
##### blocks.10.attn.proj.bias
False
##### blocks.10.norm2.weight
False
##### blocks.10.norm2.bias
False
##### blocks.10.mlp.fc1.weight
False
##### blocks.10.mlp.fc1.bias
False
##### blocks.10.mlp.fc2.weight
False
##### blocks.10.mlp.fc2.bias
False
##### blocks.11.norm1.weight
False
##### blocks.11.norm1.bias
False
##### blocks.11.attn.qkv.weight
False
##### blocks.11.attn.qkv.bias
False
##### blocks.11.attn.proj.weight
False
##### blocks.11.attn.proj.bias
False
##### blocks.11.norm2.weight
False
##### blocks.11.norm2.bias
False
##### blocks.11.mlp.fc1.weight
False
##### blocks.11.mlp.fc1.bias
False
##### blocks.11.mlp.fc2.weight
False
##### blocks.11.mlp.fc2.bias
False
##### norm.weight
False
##### norm.bias
False


##### cls_token
False
##### pos_embed
False
##### prompt_embeddings
##### deep_prompt_embeddings
##### prompt_soft_tokens_mask_cls_token
##### prompt_soft_tokens_pieces_mask_cls_token
##### patch_embed.proj.weight
False
##### patch_embed.proj.bias
False
##### blocks.0.norm1.weight
False
##### blocks.0.norm1.bias
False
##### blocks.0.attn.deep_QKV_embeddings
##### blocks.0.attn.qkv.weight
False
##### blocks.0.attn.qkv.bias
False
##### blocks.0.attn.proj.weight
False
##### blocks.0.attn.proj.bias
False
##### blocks.0.norm2.weight
False
##### blocks.0.norm2.bias
False
##### blocks.0.mlp.fc1.weight
False
##### blocks.0.mlp.fc1.bias
False
##### blocks.0.mlp.fc2.weight
False
##### blocks.0.mlp.fc2.bias
False
##### blocks.1.norm1.weight
False
##### blocks.1.norm1.bias
False
##### blocks.1.attn.deep_QKV_embeddings
##### blocks.1.attn.qkv.weight
False
##### blocks.1.attn.qkv.bias
False
##### blocks.1.attn.proj.weight
False
##### blocks.1.attn.proj.bias
False
##### blocks.1.norm2.weight
False
##### blocks.1.norm2.bias
False
##### blocks.1.mlp.fc1.weight
False
##### blocks.1.mlp.fc1.bias
False
##### blocks.1.mlp.fc2.weight
False
##### blocks.1.mlp.fc2.bias
False
##### blocks.2.norm1.weight
False
##### blocks.2.norm1.bias
False
##### blocks.2.attn.deep_QKV_embeddings
##### blocks.2.attn.qkv.weight
False
##### blocks.2.attn.qkv.bias
False
##### blocks.2.attn.proj.weight
False
##### blocks.2.attn.proj.bias
False
##### blocks.2.norm2.weight
False
##### blocks.2.norm2.bias
False
##### blocks.2.mlp.fc1.weight
False
##### blocks.2.mlp.fc1.bias
False
##### blocks.2.mlp.fc2.weight
False
##### blocks.2.mlp.fc2.bias
False
##### blocks.3.norm1.weight
False
##### blocks.3.norm1.bias
False
##### blocks.3.attn.deep_QKV_embeddings
##### blocks.3.attn.qkv.weight
False
##### blocks.3.attn.qkv.bias
False
##### blocks.3.attn.proj.weight
False
##### blocks.3.attn.proj.bias
False
##### blocks.3.norm2.weight
False
##### blocks.3.norm2.bias
False
##### blocks.3.mlp.fc1.weight
False
##### blocks.3.mlp.fc1.bias
False
##### blocks.3.mlp.fc2.weight
False
##### blocks.3.mlp.fc2.bias
False
##### blocks.4.norm1.weight
False
##### blocks.4.norm1.bias
False
##### blocks.4.attn.deep_QKV_embeddings
##### blocks.4.attn.qkv.weight
False
##### blocks.4.attn.qkv.bias
False
##### blocks.4.attn.proj.weight
False
##### blocks.4.attn.proj.bias
False
##### blocks.4.norm2.weight
False
##### blocks.4.norm2.bias
False
##### blocks.4.mlp.fc1.weight
False
##### blocks.4.mlp.fc1.bias
False
##### blocks.4.mlp.fc2.weight
False
##### blocks.4.mlp.fc2.bias
False
##### blocks.5.norm1.weight
False
##### blocks.5.norm1.bias
False
##### blocks.5.attn.deep_QKV_embeddings
##### blocks.5.attn.qkv.weight
False
##### blocks.5.attn.qkv.bias
False
##### blocks.5.attn.proj.weight
False
##### blocks.5.attn.proj.bias
False
##### blocks.5.norm2.weight
False
##### blocks.5.norm2.bias
False
##### blocks.5.mlp.fc1.weight
False
##### blocks.5.mlp.fc1.bias
False
##### blocks.5.mlp.fc2.weight
False
##### blocks.5.mlp.fc2.bias
False
##### blocks.6.norm1.weight
False
##### blocks.6.norm1.bias
False
##### blocks.6.attn.deep_QKV_embeddings
##### blocks.6.attn.qkv.weight
False
##### blocks.6.attn.qkv.bias
False
##### blocks.6.attn.proj.weight
False
##### blocks.6.attn.proj.bias
False
##### blocks.6.norm2.weight
False
##### blocks.6.norm2.bias
False
##### blocks.6.mlp.fc1.weight
False
##### blocks.6.mlp.fc1.bias
False
##### blocks.6.mlp.fc2.weight
False
##### blocks.6.mlp.fc2.bias
False
##### blocks.7.norm1.weight
False
##### blocks.7.norm1.bias
False
##### blocks.7.attn.deep_QKV_embeddings
##### blocks.7.attn.qkv.weight
False
##### blocks.7.attn.qkv.bias
False
##### blocks.7.attn.proj.weight
False
##### blocks.7.attn.proj.bias
False
##### blocks.7.norm2.weight
False
##### blocks.7.norm2.bias
False
##### blocks.7.mlp.fc1.weight
False
##### blocks.7.mlp.fc1.bias
False
##### blocks.7.mlp.fc2.weight
False
##### blocks.7.mlp.fc2.bias
False
##### blocks.8.norm1.weight
False
##### blocks.8.norm1.bias
False
##### blocks.8.attn.deep_QKV_embeddings
##### blocks.8.attn.qkv.weight
False
##### blocks.8.attn.qkv.bias
False
##### blocks.8.attn.proj.weight
False
##### blocks.8.attn.proj.bias
False
##### blocks.8.norm2.weight
False
##### blocks.8.norm2.bias
False
##### blocks.8.mlp.fc1.weight
False
##### blocks.8.mlp.fc1.bias
False
##### blocks.8.mlp.fc2.weight
False
##### blocks.8.mlp.fc2.bias
False
##### blocks.9.norm1.weight
False
##### blocks.9.norm1.bias
False
##### blocks.9.attn.deep_QKV_embeddings
##### blocks.9.attn.qkv.weight
False
##### blocks.9.attn.qkv.bias
False
##### blocks.9.attn.proj.weight
False
##### blocks.9.attn.proj.bias
False
##### blocks.9.norm2.weight
False
##### blocks.9.norm2.bias
False
##### blocks.9.mlp.fc1.weight
False
##### blocks.9.mlp.fc1.bias
False
##### blocks.9.mlp.fc2.weight
False
##### blocks.9.mlp.fc2.bias
False
##### blocks.10.norm1.weight
False
##### blocks.10.norm1.bias
False
##### blocks.10.attn.deep_QKV_embeddings
##### blocks.10.attn.qkv.weight
False
##### blocks.10.attn.qkv.bias
False
##### blocks.10.attn.proj.weight
False
##### blocks.10.attn.proj.bias
False
##### blocks.10.norm2.weight
False
##### blocks.10.norm2.bias
False
##### blocks.10.mlp.fc1.weight
False
##### blocks.10.mlp.fc1.bias
False
##### blocks.10.mlp.fc2.weight
False
##### blocks.10.mlp.fc2.bias
False
##### blocks.11.norm1.weight
False
##### blocks.11.norm1.bias
False
##### blocks.11.attn.deep_QKV_embeddings
##### blocks.11.attn.qkv.weight
False
##### blocks.11.attn.qkv.bias
False
##### blocks.11.attn.proj.weight
False
##### blocks.11.attn.proj.bias
False
##### blocks.11.norm2.weight
False
##### blocks.11.norm2.bias
False
##### blocks.11.mlp.fc1.weight
False
##### blocks.11.mlp.fc1.bias
False
##### blocks.11.mlp.fc2.weight
False
##### blocks.11.mlp.fc2.bias
False
##### fc_norm.weight
False
##### fc_norm.bias
False