import matplotlib.pyplot as plt
from math import ceil
import torch
from torchvision.transforms import ToPILImage
from collections import Counter
import circuitsvis as cv
from PIL import Image


def plot_data(data, class_images, data_id=1, dy=0.05, show_tokens=True,
              xlabel='layer', ylabel='prob', back_color=(1, 1, 1),
              class_images_line=False, color_line=False, start_layer=0):
    n = ceil(len(class_images) ** 0.5)
    f, axarr = plt.subplots(n, n, dpi=150, sharex=True, sharey=True)
    for i in range(n):
        for j in range(n):
            if n > 1:
                ax = axarr[i, j]
            else:
                ax = axarr
            if class_images_line:
                for k, cl in enumerate(class_images):
                    probs = [point[data_id][k].item() for point in
                             data[i * n + j]]
                    if color_line:
                        ax.plot(range(start_layer, start_layer + len(probs)),
                                probs, label=cl,
                                color=cl)
                    else:
                        ax.plot(range(start_layer, start_layer + len(probs)),
                                probs, label=cl)
                    ax.legend(prop={'size': 6})
            else:
                probs = [point[data_id] for point in data[i * n + j]]
                ax.plot(range(start_layer, start_layer + len(probs)), probs)

            ax.set_facecolor(back_color)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.set_title(class_images[i * n + j], fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=5)
            ax.label_outer()
            if show_tokens:
                for k in range(6, len(data[0]), 5):
                    token, _, _, _ = data[i * n + j][k]
                    y = data[i * n + j][k][data_id]
                    dx = 2
                    dy = dy
                    ax.text(k - dx, y + dy, token, fontsize=6)


def plot_data_attention(data_attention, class_images, label="image",
                        xlabel='layer', ylabel='prob', start_layer=0):
    n = ceil(len(class_images) ** 0.5)
    f, axarr = plt.subplots(n, n, dpi=150, sharex=True, sharey=True)
    d = {
        "image": 0,
        "text": 1,
        "cross": 2,
    }
    for i in range(n):
        for j in range(n):
            if n > 1:
                ax = axarr[i, j]
            else:
                ax = axarr
            probs = [point[d[label]] for point in data_attention[i * n + j]]
            ax.plot(range(start_layer, len(probs) + start_layer), probs,
                    label=label)
            ax.legend(prop={'size': 6})

            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.set_title(class_images[i * n + j], fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=5)
            ax.label_outer()


def plot_images(images):
    n = ceil(len(images) ** 0.5)
    f, axarr = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            if n > 1:
                ax = axarr[i, j]
            else:
                ax = axarr
            ax.imshow(images[i * n + j])


def get_image_tensor(atteintion):
    atteintion /= torch.max(atteintion.flatten()).item()
    atteintion *= 255
    atteintion = torch.concat(
        [atteintion.unsqueeze(0), torch.zeros((2, *atteintion.shape))],
        dim=0).to(dtype=torch.uint8)

    return atteintion


def plot_attention_mask_cache(model_cache, composite_attention, n_i, idx_word,
                              idx_image, last_layer=32, start_layer=0,
                              composite_layer=15):
    mask_mean = torch.zeros((n_i, n_i))
    max_idx = []

    w = ceil(last_layer ** 0.5)
    f, axarr = plt.subplots(w, w, dpi=150, sharex=True, sharey=True)
    for i in range(start_layer, last_layer):
        attention_pattern = model_cache["pattern", i, "attn"]

        img_attention = attention_pattern[0, :, idx_word,
                        idx_image:(idx_image + n_i * n_i)].resize(
            attention_pattern.shape[1], n_i, n_i)
        img_attention_mean = torch.mean(img_attention, dim=0)
        max_idx.append(torch.argmax(img_attention_mean).item())
        if i not in (0, 1, last_layer):
            mask_mean += img_attention_mean

        img_attention_mean = get_image_tensor(img_attention_mean)

        if i == composite_layer:
            composite_attention.append(img_attention_mean)

        mask_img = ToPILImage()(img_attention_mean)

        axarr[i // w, i % w].imshow(mask_img)

    return max_idx, mask_mean


def plot_attention_mask(model, prompt, images, start_layer=0,
                        composite_layer=15, last_layer=32,
                        attention_token="last"):
    super_pixels = []
    composite_attention = []

    model_str_tokens = model.to_str_tokens(prompt)
    idx_image = model_str_tokens.index("<image>")
    n = len(model_str_tokens)

    for img in images:
        _, model_cache = model.run_with_cache(prompt,
                                              input_images=img,
                                              remove_batch_dim=False)

        n_a = model_cache["pattern", start_layer, "attn"].shape[-1]
        n_i = int((n_a - n + 1) ** 0.5)

        model_vision_tokens = [f"i{i}{j}" for i in range(n_i) for j in
                               range(n_i)]
        model_tokens = model_str_tokens[
                       :idx_image] + model_vision_tokens + model_str_tokens[
                                                           (idx_image + 1):]
        idx_word = -1
        if attention_token != "last":
            idx_word = model_tokens.index(attention_token)

        max_idx, mask_mean = plot_attention_mask_cache(model_cache,
                                                       composite_attention,
                                                       n_i,
                                                       idx_word,
                                                       idx_image,
                                                       last_layer=last_layer,
                                                       start_layer=start_layer,
                                                       composite_layer=composite_layer)

        super_pixels.append(Counter(max_idx).most_common())

        mask_mean /= last_layer
        mask_mean = get_image_tensor(mask_mean)

        mask_img = ToPILImage()(mask_mean)
        plt.imshow(mask_img)
        plt.show()

    return composite_attention, super_pixels


def plot_text_attention(model, prompt, img, layer=15):
    model_str_tokens = model.to_str_tokens(prompt)
    idx_image = model_str_tokens.index("<image>")
    n = len(model_str_tokens)

    _, model_cache = model.run_with_cache(prompt,
                                          input_images=img,
                                          remove_batch_dim=False)
    n_a = model_cache["pattern", 0, "attn"].shape[-1]
    n_i = int((n_a - n + 1) ** 0.5)

    attention_pattern = model_cache["pattern", layer, "attn"]
    text_attention = attention_pattern[0, :, (idx_image + n_i * n_i):,
                     (idx_image + n_i * n_i):]

    return cv.attention.attention_patterns(
        tokens=model_str_tokens[(idx_image + 1):], attention=text_attention)


def plot_composite_attention(composite_attention, images):
    for i, mask_img in enumerate(composite_attention):
        img_for_mask = Image.new("L", images[i].size, 100)
        mask_img = torch.nn.functional.interpolate(mask_img.unsqueeze(1),
                                                   size=images[i].size[
                                                        ::-1]).squeeze()
        mask_img = ToPILImage()(mask_img)
        plt.imshow(Image.composite(images[i], mask_img, img_for_mask))
        plt.show()
