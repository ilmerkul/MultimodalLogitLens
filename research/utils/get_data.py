import torch


def get_data_image_cache(model, model_cache, prompt,
                         token_class_images,
                         last_layer=32,
                         start_layer=0,
                         probs_class=False):
    last_resid = model_cache[f"blocks.{last_layer - 1}.hook_resid_post"]
    last_resid = last_resid[0, -1, :]

    data_image = []
    data_attention_image = []
    for i in range(start_layer, last_layer):
        resid = model_cache[f"blocks.{i}.hook_resid_post"]
        model_logits = model(resid, start_at_layer=last_layer)[0, -1, :]
        token = model.to_single_str_token(
            torch.argmax(model_logits).item())
        probs = torch.nn.functional.softmax(model_logits)
        entropy = torch.sum(-probs * torch.log(probs)).item()
        if probs_class:
            probs = probs[token_class_images]
        else:
            probs = torch.max(probs, dim=-1)[0].item()
        cos_dist = (last_resid @ resid[0, -1,
                                 :].T).item() / torch.linalg.norm(
            last_resid, ord=2) / torch.linalg.norm(resid[0, -1, :], ord=2)
        data_image.append((token, probs, entropy, cos_dist))

        attention_pattern = model_cache["pattern", i, "attn"][0, ...]
        attention_pattern = torch.mean(attention_pattern, dim=0)

        model_str_tokens = model.to_str_tokens(prompt)
        idx_image_token = model_str_tokens.index("<image>")
        n_str = len(model_str_tokens)
        n_a = attention_pattern.shape[-1]
        idx_end_image = (n_a - n_str + 1) + idx_image_token

        attention_image = attention_pattern[idx_image_token:idx_end_image,
                          idx_image_token:idx_end_image]
        attention_text = attention_pattern[idx_end_image:, idx_end_image:]
        attention_cross_image_text = attention_pattern[idx_end_image:,
                                     idx_image_token:idx_end_image]

        attention_image = torch.mean(attention_image).item()
        attention_text = torch.mean(attention_text).item()
        attention_cross_image_text = torch.mean(
            attention_cross_image_text).item()

        data_attention_image.append(
            (attention_image, attention_text, attention_cross_image_text))

    return data_image, data_attention_image


def get_data_images(model, class_images, prompt,
                    images, last_layer=32, start_layer=0, probs_class=False):
    data = []
    data_attention = []
    token_class_images = [model.to_single_token(cl) for cl in class_images]

    for image in images:
        _, model_cache = model.run_with_cache(prompt,
                                              input_images=image,
                                              remove_batch_dim=False)

        data_image, data_attention_image = get_data_image_cache(model,
                                                                model_cache,
                                                                prompt,
                                                                token_class_images,
                                                                last_layer,
                                                                start_layer,
                                                                probs_class)

        data.append(data_image)
        data_attention.append(data_attention_image)

    return data, data_attention
