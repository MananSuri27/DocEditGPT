import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, processor, image_path, max_patches=1024):
        self.data = data
        self.processor = processor
        self.ignore_id = -100
        self.prompt_end_token = ""
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )
        self.image_path = image_path
        self.max_patches = max_patches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        command_output = data["command"]
        user_input = data["user_request"]

        top = int(data["top"])
        left = int(data["left"])
        width = int(data["width"])
        height = int(data["height"])

        image_file = data["image"]
        image_file = self.image_path + "Images/" + image_file
        image = Image.open(image_file).convert("RGB")

        encoding, bbox, bbox_header, img_sizes, images_ = self.processor.image_processor(
            images=image,
            return_tensors="pt",
            max_patches=self.max_patches,
            header_text = user_input,
            bbox_og = [top, left, height, width]
        )

        bbox =  bbox[0]
        bbox_header = bbox_header[0]
        shape = img_sizes[0][:2]

        ground_truth_mask = torch.zeros((3, shape[0], shape[1]))
    

        ground_truth_mask[1, bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]] = 1
        ground_truth_mask[2, bbox_header[0]:bbox_header[0]+bbox_header[2], bbox_header[1]:bbox_header[1]+bbox_header[3]] = 1
        ground_truth_mask = ground_truth_mask.to(bool)
        ground_truth_mask[0] = (~(ground_truth_mask[1] | ground_truth_mask[2])).to(torch.int)
        ground_truth_mask = ground_truth_mask.to(torch.float)



        encoding = {k: v.squeeze() for k, v in encoding.items()}


        encoding["text"] = f"<s_question>{user_input}</s_question> <s_answer>"
        encoding["output"] = command_output
        encoding["idx"] = idx
        encoding["shape"] = shape
        encoding["height"] = height
        encoding["ground_truth_mask"] = ground_truth_mask
        encoding["image"] = images_[0]
        encoding["bbox"] = bbox
        return encoding
