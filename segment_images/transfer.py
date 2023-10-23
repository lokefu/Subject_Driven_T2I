def transfer_masked_image_to_white_background(image_path):
    image_path = os.path.join(image_root, class_name, image_name)
    raw_image = Image.open(image_path)
    w, h = raw_image.size
    print('raw image size')
    print(raw_image.size)
    input_points = [[[w // 2, h // 2]]]
    # print(raw_image)
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    # masks = mask_generator.generate(image_name)
    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores

    print(len(masks))
    print(masks[0].shape)
    # print(type(masks))
    # print(masks.unique())
    print(masks[0].unique())
    print('scores:', scores)
    input_masks = masks[0].squeeze(0).numpy()
    outputs = make_background_white(np. array(raw_image), input_masks)
    return outputs[0]
