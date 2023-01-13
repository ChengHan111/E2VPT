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