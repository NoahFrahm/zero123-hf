import os
import argparse
import torch
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers.utils import load_image
from gradio_new import preprocess_image, create_carvekit_interface
import numpy as np
from PIL import Image
import json
import logging


def setup_global_logger(log_file):
    logger = logging.getLogger("global_logger")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Zero1to3 Inference Pipeline")
    
    parser.add_argument(
        '--gpu_index', 
        type=int, 
        required=True, 
        help='GPU index to use (e.g., 0 for cuda:0)'
    )
    
    parser.add_argument(
        '--image_paths', 
        type=str, 
        nargs='+', 
        required=True, 
        help='Paths to input images'
    )

    parser.add_argument(
        '--dest_image_names', 
        type=str, 
        nargs='+', 
        required=True, 
        help='Indices for destination images'
    )
    
    parser.add_argument(
        '--poses', 
        type=float, 
        nargs='+', 
        required=True, 
        help='List of poses as x y z per image. For example: --poses x1 y1 z1 x2 y2 z2 ...'
    )
    
    parser.add_argument(
        '--num_images_per_prompt',
        type=int,
        default=1,
        help='Number of images to generate per prompt (default: 4)'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default="logs",
        help='Directory to save generated images (default: logs)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default="kxic/stable-zero123",
        help='Model to run'
    )
    
    args = parser.parse_args()
    
    # Validate poses
    if len(args.poses) != len(args.image_paths) * 3:
        parser.error(
            f"Number of pose values ({len(args.poses)}) does not match number of images "
            f"({len(args.image_paths)}) multiplied by 3."
        )
    
    # Organize poses into list of [x, y, z]
    args.poses = [args.poses[i:i+3] for i in range(0, len(args.poses), 3)]
    
    return args


def load_model(gpu_index, logger, model_id="kxic/stable-zero123"):
    """
    Loads the Zero1to3StableDiffusionPipeline model onto the specified GPU.
    """

    # "kxic/stable-zero123"
    # "kxic/zero123-165000"
    # ashawkey/zero123-xl-diffusers
    # model_id = "kxic/zero123-xl" # zero123-105000, zero123-165000, zero123-xl, stable-zero123

    try:
        logger.info(f"Loading model '{model_id}' on GPU:{gpu_index}...")
        pipe = Zero1to3StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        
        pipe.stable_zero123 = "stable" in model_id  # Configure based on model_id

        # Enable various optimizations
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        
        # Move the model to the specified GPU
        device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        logger.info("Model loaded and moved to device successfully.")
        return pipe
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise


def run_inference(pipe, image_paths, poses, out_names, logger, target_resolution=(512,512), num_images_per_prompt=1, guidance_scale=3.0, inference_steps=50, log_dir="logs"):
    """
    Runs the inference pipeline on the provided images and poses.

    Args:
        pipe (Zero1to3StableDiffusionPipeline): The loaded model pipeline.
        image_paths (list of str): Paths to input images.
        poses (list of list of float): List of poses corresponding to each image.
        num_images_per_prompt (int): Number of images to generate per prompt.
        log_dir (str): Directory to save the generated images.
    """
    
    try:
        # Initialize Carvekit interface
        logger.info("Instantiating Carvekit HiInterface...")
        models = {'carvekit': create_carvekit_interface()}
        
        pre_images = []
        heights = []
        widths = []
        
        # Load and preprocess images
        logger.info("Loading and preprocessing images...")
        for img_path in image_paths:
            raw_im = load_image(img_path)
            input_im = preprocess_image(models, raw_im, True)
            H, W = input_im.shape[:2]
            heights.append(H)
            widths.append(W)
            pre_images.append(Image.fromarray((input_im * 255.0).astype(np.uint8)))
        
        # Ensure all images have the same dimensions
        if len(set(heights)) != 1 or len(set(widths)) != 1:
            raise ValueError("All preprocessed images must have the same dimensions.")
        
        H, W = heights[0], widths[0]
        logger.info(f"All images resized to (Height: {H}, Width: {W}).")
        
        # Run inference
        logger.info("Running inference pipeline...")
        images = pipe(
            input_imgs=pre_images, 
            prompt_imgs=pre_images, 
            poses=poses, 
            height=H, 
            width=W,
            guidance_scale=guidance_scale, 
            num_images_per_prompt=num_images_per_prompt, 
            num_inference_steps=inference_steps
        ).images
        
        # Save generated images
        logger.info(f"Saving generated images to '{log_dir}' directory...")
        os.makedirs(log_dir, exist_ok=True)
        batch_size = len(pre_images)
        image_index = 0
        for obj_idx in range(batch_size):
            for img_num in range(num_images_per_prompt):
                if image_index >= len(images):
                    break

                save_path = os.path.join(log_dir, f"{out_names[image_index]}")

                img = images[image_index]
                img = img.resize(target_resolution, resample=Image.Resampling.LANCZOS)
                img.save(save_path)
                
                logger.info(f"Saved image: {save_path}")
                image_index += 1
        logger.info("All generated images have been saved successfully.")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main():
    """
    Main function to execute the inference pipeline.
    """

    # Parse command-line arguments
    args = parse_arguments()
    gpu_index = args.gpu_index
    image_paths = args.image_paths
    poses = args.poses
    num_images_per_prompt = args.num_images_per_prompt
    log_dir = args.out_dir
    out_names = args.dest_image_names

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'generation.log')
    logger = setup_global_logger(log_file)

    if torch.cuda.is_available():
        logger.info(f"Number of CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("No CUDA devices available.")

    
    logger.info("Starting the Zero1to3 Inference Pipeline...")
    logger.info(f"Using GPU index: {gpu_index}")
    logger.info(f"Number of input images: {len(image_paths)}")
    logger.info(f"Number of poses: {len(poses)}")
    
    # Load the model
    pipe = load_model(gpu_index, logger, model_id=args.model)
    
    # Run inference
    run_inference(pipe, image_paths, poses, out_names, logger, num_images_per_prompt=num_images_per_prompt, guidance_scale=3.0, inference_steps=50, log_dir=log_dir)
    
    logger.info("Inference pipeline completed successfully.")


if __name__ == "__main__":
    main()