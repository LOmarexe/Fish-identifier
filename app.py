"""
app.py — FishNet demo
──────────────────────
Lets instructors upload or take a photo of a fish and get predictions
from either the MobileNet or ResNet model via a dropdown.

Install dependencies:
    pip install gradio torch torchvision pillow

Run:
    python app.py
Then open the printed local URL in a browser (or share the public link).
"""

import json, os, time
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ─────────────────────────────────────────────
MODELS_DIR        = "saved_models"
CONFIDENCE_THRESH = 0.3
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOX_COLOURS       = ["#FF4444", "#4488FF", "#44CC88", "#FFAA00", "#CC44FF", "#00CCCC"]
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

# ── load label map ────────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "labels.json")) as f:
    family_to_idx = json.load(f)

idx_to_family = {v: k for k, v in family_to_idx.items()}
NUM_CLASSES   = len(family_to_idx) + 1


# ── model factory ─────────────────────────────────────────────────────────────
def build_model(backbone: str) -> torch.nn.Module:
    if backbone == "mobilenet":
        m    = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        path = os.path.join(MODELS_DIR, "mobilenet_fishdet.pth")
    else:
        m    = fasterrcnn_resnet50_fpn(weights=None)
        path = os.path.join(MODELS_DIR, "resnet_fishdet.pth")

    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m


# ── pre-load both models at startup so switching is instant ──────────────────
print("loading models…")
MODELS = {
    "MobileNet v1 (faster)": build_model("mobilenet"),
    "ResNet v2 (more accurate)": build_model("resnet"),
}
print("both models ready")


# ── inference ─────────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image, model_choice: str):
    if pil_image is None:
        return None, "no image provided"

    model = MODELS[model_choice]

    # resize for display but keep original for annotation
    display_img = pil_image.convert("RGB").resize((800, 800))
    tensor      = transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(tensor)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    mask   = preds["scores"] >= CONFIDENCE_THRESH
    boxes  = preds["boxes"][mask].cpu().numpy()
    labels = preds["labels"][mask].cpu().numpy()
    scores = preds["scores"][mask].cpu().numpy()

    # ── draw boxes ────────────────────────────────────────────────────────────
    annotated = display_img.copy()
    draw      = ImageDraw.Draw(annotated)

    try:
        font       = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font       = ImageFont.load_default()
        small_font = font

    unique_families = list(dict.fromkeys(
        idx_to_family.get(int(l), f"class {l}") for l in labels
    ))
    family_colour = {f: BOX_COLOURS[i % len(BOX_COLOURS)] for i, f in enumerate(unique_families)}

    results_text = []
    for box, lbl, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        family = idx_to_family.get(int(lbl), f"class {lbl}")
        colour = family_colour[family]

        # draw box (3px border)
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                           outline=colour)

        # label background + text
        label_text = f"{family}  {score:.0%}"
        bbox_text  = draw.textbbox((x1, y1 - 26), label_text, font=font)
        draw.rectangle([bbox_text[0] - 4, bbox_text[1] - 2,
                        bbox_text[2] + 4, bbox_text[3] + 2], fill=colour)
        draw.text((x1, y1 - 26), label_text, fill="white", font=font)

        results_text.append(f"**{family}** — {score:.1%} confidence")

    # ── summary text ──────────────────────────────────────────────────────────
    if len(boxes) == 0:
        summary = (f"**No fish detected** above {CONFIDENCE_THRESH:.0%} confidence.\n\n"
                   f"Try lowering the threshold or use a clearer image.\n\n"
                   f"*Inference: {elapsed_ms:.0f} ms using {model_choice}*")
    else:
        detections = "\n".join(f"- {r}" for r in results_text)
        summary = (f"### Detected {len(boxes)} fish\n\n"
                   f"{detections}\n\n"
                   f"*Inference: {elapsed_ms:.0f} ms using {model_choice}*")

    return annotated, summary


# ── gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="FishNet — fish family detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# FishNet — fish family detector")
    gr.Markdown(
        "Upload a photo or use your camera to identify the fish family. "
        "Switch between the two trained models using the dropdown."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Input image",
                type="pil",
                sources=["upload", "webcam"],
            )
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="MobileNet v1 (faster)",
                label="Model",
                info="MobileNet is faster; ResNet is more accurate"
            )
            run_btn = gr.Button("Identify fish", variant="primary")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Annotated output", type="pil")
            text_output  = gr.Markdown(label="Results")

    # examples — replace paths with real images from your val set if available
    gr.Examples(
        examples=[],           # add e.g. [["examples/trout.jpg", "MobileNet v1 (faster)"]]
        inputs=[image_input, model_choice],
    )

    run_btn.click(
        fn=predict,
        inputs=[image_input, model_choice],
        outputs=[image_output, text_output],
    )

    # also run when model dropdown changes (so instructors can switch and see instantly)
    model_choice.change(
        fn=predict,
        inputs=[image_input, model_choice],
        outputs=[image_output, text_output],
    )

    gr.Markdown(
        "---\n"
        "*FishNet — built with PyTorch + Gradio · "
        "FishNet dataset (Khan et al.) · "
        "Backbone: Faster R-CNN*"
    )


if __name__ == "__main__":
    demo.launch(
        share=True,        # generates a public gradio.live link valid for 72 h
        show_error=True,
    )