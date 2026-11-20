import torch
from torch.nn import functional as F


def extract_patch_tokens(visual, images):
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = x + visual.positional_embedding.to(x.dtype)[1:, :]
    return x


def generate_stage_masks(grid_h, grid_w, scale, stride):
    row_starts = list(range(0, grid_h - scale + 1, stride))
    if row_starts[-1] != grid_h - scale:
        row_starts.append(grid_h - scale)

    col_starts = list(range(0, grid_w - scale + 1, stride))
    if col_starts[-1] != grid_w - scale:
        col_starts.append(grid_w - scale)

    index_grid = torch.arange(grid_h * grid_w, dtype=torch.int64).reshape(grid_h, grid_w)
    masks = []
    for r in row_starts:
        for c in col_starts:
            window = index_grid[r:r + scale, c:c + scale].reshape(-1)
            masks.append(window)
    return masks


def run_selected_windows(visual, patch_tokens_single, selected_masks):
    device = patch_tokens_single.device
    num_windows = len(selected_masks)

    index_tensor = torch.stack(selected_masks, dim=0).to(device)
    flat_tokens = patch_tokens_single[0]
    mx = flat_tokens[index_tensor]

    cls_token = visual.class_embedding.to(mx.dtype) + visual.positional_embedding.to(mx.dtype)[0, :]
    cls_tokens = cls_token.reshape(1, 1, -1).expand(num_windows, 1, -1)
    mx = torch.cat((cls_tokens, mx), dim=1)

    mx = visual.patch_dropout(mx)
    mx = visual.ln_pre(mx)

    mx = mx.permute(1, 0, 2)
    mx = visual.transformer(mx)
    mx = mx.permute(1, 0, 2)

    if visual.attn_pool is not None:
        mx = visual.attn_pool(mx)
        mx = visual.ln_post(mx)
        pooled, _ = visual._global_pool(mx)
    else:
        pooled, _ = visual._global_pool(mx)
        pooled = visual.ln_post(pooled)

    if visual.proj is not None:
        pooled = pooled @ visual.proj

    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled


@torch.no_grad()
def forward_hierarchical(winclip_model, images, scales=(4, 3, 2), keep_ratios=(1.0, 0.5, 0.35), first_stride=2):
    if winclip_model.precision == "fp16":
        images = images.half()

    visual = winclip_model.model.visual
    grid_h, grid_w = winclip_model.grid_size
    num_patches = grid_h * grid_w
    text_features = winclip_model.text_features

    all_patch_tokens = extract_patch_tokens(visual, images)
    batch_size = images.shape[0]

    am_np_list = []
    total_windows = 0

    for b in range(batch_size):
        patch_tokens = all_patch_tokens[b:b + 1]
        coarse_map = None
        guidance_map = None
        refined_sum = torch.zeros(num_patches)
        refined_count = torch.zeros(num_patches)

        for stage_idx in range(len(scales)):
            scale = scales[stage_idx]
            ratio = keep_ratios[stage_idx]
            stride = first_stride if stage_idx == 0 else 1

            candidate_masks = generate_stage_masks(grid_h, grid_w, scale, stride)

            if stage_idx == 0 or ratio >= 1.0:
                selected_masks = candidate_masks
            else:
                # Rank candidate windows by average anomaly score inside each window
                window_scores = []
                for mask in candidate_masks:
                    window_scores.append(guidance_map[mask].mean().item())

                num_keep = max(1, int(len(candidate_masks) * ratio))
                sorted_indices = sorted(range(len(window_scores)), key=lambda i: window_scores[i], reverse=True)
                selected_masks = [candidate_masks[i] for i in sorted_indices[:num_keep]]

            total_windows += len(selected_masks)

            features = run_selected_windows(visual, patch_tokens, selected_masks)
            probs = (100.0 * features @ text_features.T).softmax(dim=-1)
            normality = probs[:, 0].float().cpu().clamp(min=1e-6)
            inv_norm = 1.0 / normality

            token_sum = torch.zeros(num_patches)
            token_weight = torch.zeros(num_patches)
            for w_idx, mask in enumerate(selected_masks):
                token_sum[mask] += inv_norm[w_idx]
                token_weight[mask] += 1.0

            covered = token_weight > 0
            stage_values = token_sum[covered] / token_weight[covered]

            if stage_idx == 0:
                coarse_map = token_sum / token_weight.clamp(min=1.0)
                guidance_map = coarse_map.clone()
                # If using only 2 scales (e.g., 3 -> 2), include Stage 0 in final score
                if len(scales) <= 2:
                    refined_sum[covered] += stage_values
                    refined_count[covered] += 1.0
            else:
                guidance_map[covered] = stage_values
                refined_sum[covered] += stage_values
                refined_count[covered] += 1.0

        # Use sharp refined stages (b x b, c x c) where evaluated; coarse radar fills skipped background
        final_inv_norm = coarse_map.clone()
        has_refined = refined_count > 0
        final_inv_norm[has_refined] = refined_sum[has_refined] / refined_count[has_refined]

        final_scores = 1.0 - 1.0 / final_inv_norm
        anomaly_map = final_scores.reshape(1, 1, grid_h, grid_w)
        anomaly_map = F.interpolate(
            anomaly_map,
            size=(winclip_model.out_size_h, winclip_model.out_size_w),
            mode="bilinear",
            align_corners=False,
        )
        am_np_list.append(anomaly_map.squeeze().cpu().numpy())

    winclip_model.last_windows_used = total_windows / batch_size
    return am_np_list
