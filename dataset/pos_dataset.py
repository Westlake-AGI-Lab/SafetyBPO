import os, json, random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PosDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        tokenizer,
        resolution: int = 512,
        random_crop: bool = False,
        no_hflip: bool = False,
        proportion_empty_prompts: float = 0.0,
        sdxl: bool = False
    ):
        if not sdxl:
            assert tokenizer is not None, "Tokenizer must be provided for non-SDXL mode"

        with open(metadata_path, 'r', encoding='utf-8') as f: 
            self.records = [json.loads(line) for line in f]

        self.image_root = image_root
        self.tokenizer = tokenizer if not sdxl else None
        self.proportion_empty = proportion_empty_prompts
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
        if random.random() < self.proportion_empty:
            caption = ""
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
        try:
            im0 = self.transform(self._load_pil(rec['preferred'])) # preferred  chosen
            im1 = self.transform(self._load_pil(rec['unpreferred'])) # unpreferred rejected
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        out = {'pixel_values': torch.cat([im0, im1], dim=0)}  # Cat to create [6, H, W] for DPO
        caption = rec.get('caption', "")
        token_or_text = self._tokenize(caption)
        out['caption' if self.sdxl else 'input_ids'] = token_or_text

        return out


def collate_fn(batch):

    out = {'pixel_values': torch.stack([ex['pixel_values'] for ex in batch]).to(memory_format=torch.contiguous_format).float()}
    if 'input_ids' in batch[0]:
        out['input_ids'] = torch.stack([ex['input_ids'] for ex in batch])
    if 'caption' in batch[0]:
        out['caption'] = [ex['caption'] for ex in batch]
    return out
