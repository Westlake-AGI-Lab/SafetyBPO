import os, json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class NegDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        tokenizer,
        resolution: int = 512,
        random_crop: bool = False,
        no_hflip: bool = False,
        sdxl: bool = False
    ):
        if not sdxl:
            assert tokenizer is not None, "Tokenizer must be provided for non-SDXL mode"

        with open(metadata_path, 'r', encoding='utf-8') as f: 
            self.records = [json.loads(line) for line in f]

        self.image_root = image_root
        self.tokenizer = tokenizer if not sdxl else None
        self.sdxl = sdxl

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution) if random_crop else transforms.CenterCrop(resolution),
            transforms.Lambda(lambda x: x) if no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.records)

    def _load_pil(self, rel_path: str) -> Image.Image:
        if rel_path is None:
            raise ValueError(f"Image path is None")
        img = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
        return img

    def _tokenize(self, caption: str):
        if self.sdxl:
            return caption
        return self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        fmt = _detect_record_format(rec)

        if fmt == "dpo":
            preferred_path = rec['preferred']
            unpreferred_path = rec['unpreferred']
            file_name = rec.get('file_name', '') or ''
            if str(file_name).endswith('_u.jpg'):
                preferred_path, unpreferred_path = unpreferred_path, preferred_path
            im0 = self.transform(self._load_pil(preferred_path))
            im1 = self.transform(self._load_pil(unpreferred_path))
            pixel_values = torch.cat([im0, im1], dim=0)
            caption = rec.get('caption', "")
            token_or_text = self._tokenize(caption)
            return {
                'pixel_values': pixel_values,
                'caption' if self.sdxl else 'input_ids': token_or_text,
                'data_format': 'dpo',
            }
        else:
            img_path = rec.get('file_name')
            if img_path is None:
                raise ValueError(f"TPO record must have 'image', 'file_name', or 'preferred'. Keys: {list(rec.keys())}")
            im = self.transform(self._load_pil(img_path))
            pixel_values = torch.cat([im, im], dim=0)
            input_ids_unsafe = self._tokenize(rec.get('unsafe_caption', ''))
            input_ids_safe = self._tokenize(rec.get('safe_caption', ''))
            return {
                'pixel_values': pixel_values,
                'input_ids_unsafe': input_ids_unsafe,
                'input_ids_safe': input_ids_safe,
                'data_format': 'tpo',
            }

def _detect_record_format(rec: dict) -> str:
    """Detect if record is DPO or TPO format."""
    has_preferred = rec.get("preferred") is not None
    has_unpreferred = rec.get("unpreferred") is not None
    has_caption = rec.get("caption") is not None and rec.get("caption") != ""
    has_safe_caption = rec.get("safe_caption") is not None and rec.get("safe_caption") != ""
    has_unsafe_caption = rec.get("unsafe_caption") is not None and rec.get("unsafe_caption") != ""
    has_image = rec.get("image") is not None
    has_file_name = rec.get("file_name") is not None

    has_tpo_image = has_image or has_file_name or (has_preferred and not has_unpreferred)
    if (has_safe_caption and has_unsafe_caption) and has_tpo_image:
        return "tpo"
    if has_preferred and has_unpreferred and has_caption:
        return "dpo"
    raise ValueError(
        f"Unknown record format. DPO needs: preferred, unpreferred, caption. "
        f"TPO needs: image/file_name/preferred (single image path), safe_caption, unsafe_caption. "
        f"Keys: {list(rec.keys())}"
    )

def collate_fn(examples):
    """Collate for mixed TPO/DPO batches."""
    valid_examples = [ex for ex in examples if ex.get("pixel_values") is not None]
    if len(valid_examples) == 0:
        raise ValueError("No valid examples in batch")

    pixel_values = torch.stack([ex["pixel_values"] for ex in valid_examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    final_dict = {"pixel_values": pixel_values}

    data_formats = [ex.get("data_format", "tpo") for ex in valid_examples]
    unique_formats = set(data_formats)

    if len(unique_formats) == 1:
        data_format = data_formats[0]
        if data_format == "tpo":
            if "caption_unsafe" in valid_examples[0]:
                final_dict["caption_unsafe"] = [ex["caption_unsafe"] for ex in valid_examples]
                final_dict["caption_safe"] = [ex["caption_safe"] for ex in valid_examples]
            else:
                final_dict["input_ids_unsafe"] = torch.stack([ex["input_ids_unsafe"] for ex in valid_examples])
                final_dict["input_ids_safe"] = torch.stack([ex["input_ids_safe"] for ex in valid_examples])
        elif data_format == "dpo":
            if "caption" in valid_examples[0]:
                final_dict["caption"] = [ex["caption"] for ex in valid_examples]
            else:
                final_dict["input_ids"] = torch.stack([ex["input_ids"] for ex in valid_examples])
        final_dict["data_format"] = data_format
    else:
        tpo_examples = [ex for ex, fmt in zip(valid_examples, data_formats) if fmt == "tpo"]
        dpo_examples = [ex for ex, fmt in zip(valid_examples, data_formats) if fmt == "dpo"]
        if tpo_examples:
            final_dict["tpo_pixel_values"] = torch.stack([ex["pixel_values"] for ex in tpo_examples]).to(memory_format=torch.contiguous_format).float()
            if "caption_unsafe" in tpo_examples[0]:
                final_dict["tpo_caption_unsafe"] = [ex["caption_unsafe"] for ex in tpo_examples]
                final_dict["tpo_caption_safe"] = [ex["caption_safe"] for ex in tpo_examples]
            else:
                final_dict["tpo_input_ids_unsafe"] = torch.stack([ex["input_ids_unsafe"] for ex in tpo_examples])
                final_dict["tpo_input_ids_safe"] = torch.stack([ex["input_ids_safe"] for ex in tpo_examples])
        if dpo_examples:
            final_dict["dpo_pixel_values"] = torch.stack([ex["pixel_values"] for ex in dpo_examples]).to(memory_format=torch.contiguous_format).float()
            if "caption" in dpo_examples[0]:
                final_dict["dpo_caption"] = [ex["caption"] for ex in dpo_examples]
            else:
                final_dict["dpo_input_ids"] = torch.stack([ex["input_ids"] for ex in dpo_examples])
        final_dict["data_format"] = "mixed"
        final_dict["format_indices"] = data_formats

    return final_dict

