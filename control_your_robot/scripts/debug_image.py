
def prepare_inputs(self, images, state, instruction="Stack the blocks"):
    import torch
    from PIL import Image
    
    # 构造文本
    text = f"<|im_start|>system\\nYou are a helpful robot assistant.<|im_end|>\\n"
    text += f"<|im_start|>user\\n<image>\\n{instruction}<|im_end|>\\n"
    text += f"<|im_start|>assistant\\n"
    
    print(f"[DEBUG] Text: {repr(text)}")
    
    # 转换图像
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))
    
    print(f"[DEBUG] Number of PIL images: {len(pil_images)}")
    
    # 检查 processor
    processor = self.model.processor
    print(f"[DEBUG] Processor type: {type(processor)}")
    print(f"[DEBUG] Processor use_fast_tokenizer: {getattr(processor, "use_fast_tokenizer", "unknown")}")
    
    # 处理
    inputs = processor(
        text=[text],
        images=pil_images if len(pil_images) > 0 else None,
        return_tensors="pt"
    )
    
    print(f"[DEBUG] Input keys: {list(inputs.keys())}")
    print(f"[DEBUG] input_ids shape: {inputs[input_ids].shape}")
    
    # 检查是否有图像 token
    image_token_id = getattr(processor, "image_token_id", None)
    print(f"[DEBUG] Image token ID: {image_token_id}")
    if "<image>" in processor.tokenizer.vocab:
        print(f"[DEBUG] <image> in vocab, ID: {processor.tokenizer.vocab["<image>"]}")
    
    # 计算 token 数
    input_ids = inputs["input_ids"]
    num_image_tokens = (input_ids == image_token_id).sum().item() if image_token_id is not None else 0
    print(f"[DEBUG] Number of image tokens in input_ids: {num_image_tokens}")
    
    if "pixel_values" in inputs:
        print(f"[DEBUG] pixel_values shape: {inputs[pixel_values].shape}")
        print(f"[DEBUG] image_grid_thw: {inputs.get(image_grid_thw, N/A)}")
    
    return inputs
