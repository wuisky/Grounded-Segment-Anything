import argparse
import json
import os
import sys
import random
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

import cv2
import matplotlib
import matplotlib.pyplot as plt
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import SamPredictor, sam_model_registry

# =====================================================
# ✅ 再現性を完全に固定するための初期化関数
# =====================================================
def set_deterministic(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

set_deterministic(0)
# =====================================================


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(phrase)
    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.default_rng(0).random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, image_size=None):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    if image_size is not None:
        mask_img_np = mask_img.numpy().astype(np.uint8)
        mask_img_pil = Image.fromarray(mask_img_np)
        mask_img_pil = mask_img_pil.resize(image_size, resample=Image.NEAREST)
        plt.figure(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
        plt.imshow(np.array(mask_img_pil), cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, "mask_resize.png"), bbox_inches="tight", pad_inches=0)

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "mask.jpg"), bbox_inches="tight", dpi=300, pad_inches=0)

    json_data = [{"value": 0, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]
        json_data.append({
            "value": value,
            "label": name,
            "logit": float(logit),
            "box": box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, "mask.json"), "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--sam_version", type=str, default="vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None)
    parser.add_argument("--use_sam_hq", action="store_true")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_pil, image = load_image(args.input_image)
    model = load_model(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, device=args.device)
    image_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))

    boxes_filt, pred_phrases = get_grounding_output(
        model, image, args.text_prompt, args.box_threshold, args.text_threshold, device=args.device
    )

    predictor = SamPredictor(
        sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(args.device)
    )
    predictor.model.eval()

    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(args.device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(args.device),
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis("off")
    plt.savefig(os.path.join(args.output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    save_mask_data(args.output_dir, masks, boxes_filt, pred_phrases, size)
