# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import torch
import ray
from PIL import Image
from diffusers import FluxPipeline


class AlchemistDataset:
    def __init__(self, data_path: str, prompt_column: str = 'prompt', key_column: str = 'img_key'):
        self.prompts = []
        self.keys = []
        data_path = Path(data_path)

        df = pd.read_csv(data_path)
        if prompt_column not in df.columns:
            raise ValueError(f"Column '{prompt_column}' not found in CSV")

        self.prompts = df[prompt_column].tolist()

        if key_column in df.columns:
            self.keys = df[key_column].tolist()
        else:
            self.keys = [f"{i:05d}" for i in range(len(self.prompts))]

        print(f"Loaded {len(self.prompts)} prompts from Alchemist dataset")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            'prompt': self.prompts[idx],
            'key': self.keys[idx],
            'index': idx
        }


class DPGDataset:
    def __init__(self, prompts_dir: str):
        self.prompts = []
        self.keys = []
        prompts_dir = Path(prompts_dir)

        if not prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {prompts_dir}")

        txt_files = sorted(prompts_dir.glob("*.txt"))

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()

            if prompt:
                self.prompts.append(prompt)
                self.keys.append(txt_file.stem)

        print(f"Loaded {len(self.prompts)} prompts from DPG-Bench dataset")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            'prompt': self.prompts[idx],
            'key': self.keys[idx],
            'index': idx
        }


class UniGenBenchDataset:
    def __init__(self, csv_path: str):
        self.prompts = []
        self.indices = []
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise ValueError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if 'index' not in df.columns or 'prompt_en' not in df.columns:
            raise ValueError(f"Required columns not found in CSV")

        self.prompts = df['prompt_en'].tolist()
        self.indices = df['index'].tolist()

        print(f"Loaded {len(self.prompts)} prompts from UniGenBench dataset")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            'prompt': self.prompts[idx],
            'key': str(self.indices[idx]),
            'index': self.indices[idx]
        }


class HPSv3Dataset:
    def __init__(self, benchmark_dir: str):
        self.prompts = []
        self.names = []
        self.categories = []
        benchmark_dir = Path(benchmark_dir)

        if not benchmark_dir.exists():
            raise ValueError(f"Benchmark directory not found: {benchmark_dir}")

        json_files = sorted(benchmark_dir.glob("benchmark_*.json"))

        if not json_files:
            raise ValueError(f"No benchmark JSON files found in {benchmark_dir}")

        for json_file in json_files:
            category = json_file.stem.replace("benchmark_", "")

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                caption = item.get('caption', '')
                image_file = item.get('image_file', '')

                if caption:
                    self.prompts.append(caption)
                    name = Path(image_file).stem
                    self.names.append(name)
                    self.categories.append(category)

        print(f"Loaded {len(self.prompts)} prompts from HPSv3 benchmark ({len(json_files)} categories)")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            'prompt': self.prompts[idx],
            'key': self.names[idx],
            'category': self.categories[idx],
            'index': idx
        }


def tile_images(images: List[Image.Image], resolution: int) -> Image.Image:
    num_images = len(images)
    resized_images = [img.resize((resolution, resolution), Image.LANCZOS) for img in images]

    if num_images == 1:
        return resized_images[0]
    elif num_images == 2:
        tiled = Image.new('RGB', (2 * resolution, resolution))
        tiled.paste(resized_images[0], (0, 0))
        tiled.paste(resized_images[1], (resolution, 0))
        return tiled
    elif num_images == 3:
        tiled = Image.new('RGB', (2 * resolution, 2 * resolution))
        tiled.paste(resized_images[0], (0, 0))
        tiled.paste(resized_images[1], (resolution, 0))
        tiled.paste(resized_images[2], (0, resolution))
        return tiled
    else:
        tiled = Image.new('RGB', (2 * resolution, 2 * resolution))
        tiled.paste(resized_images[0], (0, 0))
        tiled.paste(resized_images[1], (resolution, 0))
        tiled.paste(resized_images[2], (0, resolution))
        tiled.paste(resized_images[3], (resolution, resolution))
        return tiled


def load_checkpoint_into_transformer(transformer, checkpoint_path: str) -> None:
    """Load checkpoint into transformer model.

    Supports:
    - FAIL checkpoint: checkpoint-{step}/hf_weights/diffusion_pytorch_model.safetensors
    - SFT checkpoint: checkpoint-{step}-{epoch}/diffusion_pytorch_model.safetensors
    - Direct safetensors file
    """
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)

    print(f"Loading transformer checkpoint from {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path))
    elif checkpoint_path.is_dir():
        hf_weights_path = checkpoint_path / "hf_weights" / "diffusion_pytorch_model.safetensors"
        direct_path = checkpoint_path / "diffusion_pytorch_model.safetensors"

        if hf_weights_path.exists():
            print(f"  Loading from FAIL checkpoint format: {hf_weights_path}")
            state_dict = load_file(str(hf_weights_path))
        elif direct_path.exists():
            print(f"  Loading from SFT checkpoint format: {direct_path}")
            state_dict = load_file(str(direct_path))
        else:
            raise FileNotFoundError(
                f"No safetensors file found at {checkpoint_path}. "
                f"Expected either {hf_weights_path} or {direct_path}"
            )
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "")
        new_key = new_key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = transformer.load_state_dict(cleaned_state_dict, strict=False)

    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)}")
    print("Transformer checkpoint loaded successfully")


def suppress_pipeline_warnings():
    import logging
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=".*truncated because.*can only handle sequences.*",
        category=UserWarning,
    )

    diffusers_logger = logging.getLogger("diffusers.pipelines")
    diffusers_logger.setLevel(logging.ERROR)


@ray.remote(num_gpus=1)
class FluxGenerationWorker:
    def __init__(self, worker_id: int, args: argparse.Namespace):
        self.worker_id = worker_id
        self.args = args

        if torch.cuda.is_available():
            self.device = "cuda:0"
            torch.cuda.set_device(0)
        else:
            self.device = "cpu"

        print(f"[Worker {worker_id}] Initializing on {self.device}")

        self.pipe = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )

        if args.load_checkpoint and args.checkpoint_path:
            load_checkpoint_into_transformer(self.pipe.transformer, args.checkpoint_path)

        self.pipe = self.pipe.to(self.device)
        suppress_pipeline_warnings()
        self.pipe.set_progress_bar_config(disable=True)

        print(f"[Worker {worker_id}] Initialization complete")

    def generate_batch(
        self,
        batch_data: List[Dict[str, Any]],
        output_dir: str,
        dataset_type: str,
    ) -> List[Dict[str, Any]]:
        results = []
        output_dir = Path(output_dir)

        for data in batch_data:
            prompt = data['prompt']
            key = data['key']
            index = data['index']
            category = data.get('category')

            try:
                generator = torch.Generator(device=self.device)
                seed = self.args.seed + index
                generator.manual_seed(seed)

                with torch.no_grad():
                    output = self.pipe(
                        prompt=prompt,
                        negative_prompt=self.args.negative_prompt,
                        true_cfg_scale=self.args.true_cfg_scale,
                        guidance_scale=self.args.guidance_scale,
                        height=self.args.resolution,
                        width=self.args.resolution,
                        num_inference_steps=self.args.num_inference_steps,
                        max_sequence_length=self.args.max_sequence_length,
                        generator=generator,
                        num_images_per_prompt=self.args.num_images_per_prompt
                    )
                    images = output.images

                if dataset_type in ['alchemist', 'unigenbench', 'hpsv3']:
                    for img_idx, image in enumerate(images):
                        if dataset_type == 'alchemist':
                            key_dir = output_dir / key
                            key_dir.mkdir(parents=True, exist_ok=True)

                            if img_idx == 0:
                                prompt_file = key_dir / "prompt.txt"
                                with open(prompt_file, 'w', encoding='utf-8') as f:
                                    f.write(prompt)

                            img_path = key_dir / f"image_{img_idx}.{self.args.save_format}"
                            image.save(img_path)

                        elif dataset_type == 'unigenbench':
                            img_path = output_dir / f"{index}_{img_idx}.{self.args.save_format}"
                            image.save(img_path)

                        elif dataset_type == 'hpsv3':
                            category_dir = output_dir / category
                            category_dir.mkdir(parents=True, exist_ok=True)
                            img_path = category_dir / f"{key}_{img_idx}.{self.args.save_format}"
                            image.save(img_path)

                        result_entry = {
                            'prompt': prompt,
                            'key': key,
                            'index': index,
                            'img_idx': img_idx,
                            'img_path': str(img_path),
                            'dataset_type': dataset_type,
                        }
                        if category is not None:
                            result_entry['category'] = category
                        results.append(result_entry)
                else:
                    for img_idx, image in enumerate(images):
                        result_entry = {
                            'image': image,
                            'prompt': prompt,
                            'key': key,
                            'index': index,
                            'img_idx': img_idx,
                            'dataset_type': dataset_type,
                        }
                        if category is not None:
                            result_entry['category'] = category
                        results.append(result_entry)

                print(f"[Worker {self.worker_id}] Generated {len(images)} images for prompt index {index}")

            except Exception as e:
                print(f"[Worker {self.worker_id}] Error processing prompt '{prompt[:50]}...': {e}")
                import traceback
                traceback.print_exc()
                continue

        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Ray-based distributed FLUX image generation")

    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=['alchemist', 'dpg', 'unigenbench', 'hpsv3'],
                        help="Type of dataset to use")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset (CSV for alchemist/unigenbench, directory for dpg/hpsv3)")
    parser.add_argument("--prompt_column", type=str, default="prompt",
                        help="Column name for prompts in CSV (alchemist)")
    parser.add_argument("--key_column", type=str, default="img_key",
                        help="Column name for keys in CSV (alchemist)")
    parser.add_argument("--prompt_version", type=str, default="normal",
                        choices=['normal', 'long'],
                        help="Prompt version for unigenbench (normal or long)")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="./data/flux",
                        help="Path to pretrained FLUX model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--load_checkpoint", action="store_true",
                        help="Load checkpoint if checkpoint_path is provided")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images")
    parser.add_argument("--jsonl_output", type=str, default=None,
                        help="Path to output JSONL file (default: {output_dir}/results.jsonl)")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0,
                        help="True CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Number of inference steps")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max sequence length for text encoder")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_images_per_prompt", type=int, default=4,
                        help="Number of images per prompt")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt")
    parser.add_argument("--save_format", type=str, default="png",
                        choices=['png', 'jpg'],
                        help="Image save format")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prompts")

    parser.add_argument("--ray_address", type=str, default=None,
                        help="Ray cluster address (auto-init if None)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers (auto-detect GPUs if None)")

    return parser.parse_args()


def load_dataset(args):
    if args.dataset_type == 'alchemist':
        dataset = AlchemistDataset(args.data_path, args.prompt_column, args.key_column)
    elif args.dataset_type == 'dpg':
        dataset = DPGDataset(args.data_path)
    elif args.dataset_type == 'unigenbench':
        if Path(args.data_path).is_file():
            csv_path = args.data_path
        else:
            base_dir = Path(args.data_path)
            if args.prompt_version == 'normal':
                csv_path = base_dir / "test_prompts_en.csv"
            else:
                csv_path = base_dir / "test_prompts_en_long.csv"
        dataset = UniGenBenchDataset(csv_path)
    elif args.dataset_type == 'hpsv3':
        if Path(args.data_path).is_dir():
            benchmark_dir = args.data_path
        else:
            benchmark_dir = Path(args.data_path) / "benchmark"
            if not benchmark_dir.exists():
                benchmark_dir = args.data_path
        dataset = HPSv3Dataset(benchmark_dir)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    return dataset


def save_images_and_generate_jsonl(results: List[Dict], output_dir: Path, dataset_type: str, save_format: str) -> List[Dict]:
    jsonl_entries = []

    if dataset_type == 'alchemist':
        for result in results:
            jsonl_entries.append({
                'id': f"{result['key']}_{result['img_idx']}",
                'dataset': dataset_type,
                'prompt_index': result['index'],
                'img_path': result['img_path'],
                'prompt': result['prompt'],
            })

    elif dataset_type == 'dpg':
        key_groups = {}
        for result in results:
            key = result['key']
            if key not in key_groups:
                key_groups[key] = []
            key_groups[key].append(result)

        for key, group in key_groups.items():
            images = [r['image'] for r in group]
            tiled_image = tile_images(images, group[0]['image'].size[0])

            img_path = output_dir / f"{key}.{save_format}"
            tiled_image.save(img_path)

            for result in group:
                img_idx = result['img_idx']
                jsonl_entries.append({
                    'id': f"{key}_{img_idx}",
                    'dataset': dataset_type,
                    'prompt_index': result['index'],
                    'img_path': str(img_path),
                    'prompt': result['prompt'],
                })

    elif dataset_type == 'unigenbench':
        for result in results:
            jsonl_entries.append({
                'id': f"{result['index']}_{result['img_idx']}",
                'dataset': dataset_type,
                'prompt_index': result['index'],
                'img_path': result['img_path'],
                'prompt': result['prompt'],
            })

    elif dataset_type == 'hpsv3':
        for result in results:
            jsonl_entries.append({
                'id': f"{result.get('category', 'unknown')}_{result['key']}_{result['img_idx']}",
                'dataset': dataset_type,
                'category': result.get('category', 'unknown'),
                'prompt_index': result['index'],
                'img_path': result['img_path'],
                'prompt': result['prompt'],
            })

    return jsonl_entries


def main():
    args = parse_args()

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    if args.num_workers is None:
        available_resources = ray.available_resources()
        num_gpus = int(available_resources.get("GPU", 0))
        args.num_workers = num_gpus if num_gpus > 0 else 1

    print(f"Initializing {args.num_workers} Ray workers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.jsonl_output is None:
        args.jsonl_output = output_dir / "results.jsonl"

    print(f"Loading {args.dataset_type} dataset from {args.data_path}")
    dataset = load_dataset(args)

    if args.limit:
        dataset.prompts = dataset.prompts[:args.limit]
        if hasattr(dataset, 'keys'):
            dataset.keys = dataset.keys[:args.limit]
        if hasattr(dataset, 'indices'):
            dataset.indices = dataset.indices[:args.limit]
        if hasattr(dataset, 'names'):
            dataset.names = dataset.names[:args.limit]
        if hasattr(dataset, 'categories'):
            dataset.categories = dataset.categories[:args.limit]
        print(f"Limited to {args.limit} prompts")

    print(f"Total prompts: {len(dataset)}")
    print(f"Images per prompt: {args.num_images_per_prompt}")
    print(f"Total images to generate: {len(dataset) * args.num_images_per_prompt}")

    print("Creating Ray workers...")
    workers = [FluxGenerationWorker.remote(i, args) for i in range(args.num_workers)]

    all_data = [dataset[i] for i in range(len(dataset))]
    batch_size = len(all_data) // args.num_workers + (1 if len(all_data) % args.num_workers else 0)

    batches = []
    for i in range(args.num_workers):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(all_data))
        if start_idx < len(all_data):
            batches.append(all_data[start_idx:end_idx])
        else:
            batches.append([])

    print(f"Distributing work: {[len(b) for b in batches]} prompts per worker")

    print("Submitting generation tasks...")
    futures = []
    for worker, batch in zip(workers, batches):
        if batch:
            future = worker.generate_batch.remote(batch, str(output_dir), args.dataset_type)
            futures.append(future)

    print("Waiting for results...")
    all_results = []
    for future in futures:
        try:
            results = ray.get(future)
            all_results.extend(results)
        except Exception as e:
            print(f"Error collecting results: {e}")
            import traceback
            traceback.print_exc()

    print(f"Collected {len(all_results)} results from workers")

    print("Saving images and generating JSONL...")
    jsonl_entries = save_images_and_generate_jsonl(
        all_results, output_dir, args.dataset_type, args.save_format
    )

    with open(args.jsonl_output, 'w', encoding='utf-8') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"JSONL file saved to {args.jsonl_output}")
    print("Generation complete!")

    ray.shutdown()


if __name__ == "__main__":
    main()
