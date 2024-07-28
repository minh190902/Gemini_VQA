from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor 
import torch
from PIL import Image
import easyocr

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ImageQuestionAnswering:
    def __init__(self, device="cuda"):
        self.device = device
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.load_models()
        self.reader = easyocr.Reader(['vi'])
    
    def load_models(self):
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # QWEN 2
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-7B-Instruct",
            quantization_config=self.quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        
        # Florence 2
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    
    def perform_ocr(self, image):
        result = self.reader.readtext(image)
        ocr_text = "\n".join([res[1] for res in result])
        return "Info of text in image: " + ocr_text
    
    def perform_image(self, task_prompt, image, text_input=None):
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + "." + text_input
            
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

        return parsed_answer[task_prompt]

    
    def generate_response(self, image, user_prompt):
        # OCR Processing
        ocr_text = self.perform_ocr(image)
        print("ocr")
        print(ocr_text)
        print("-"*20)
        
        info_img = self.perform_image(task_prompt='<MORE_DETAILED_CAPTION>', image=image, text_input=None)
        
        print("florence")
        print(info_img)
        print("-"*20)
        # Prepare the full prompt for Qwen model
        PROMPT_TEMPLATE = """
        You are a helpful assistant who always responds in vietnamese!
        Some rules to follow:
            1. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
            2. You must focus on OCR task.
            3. Answer question from <User> with clarification and high precision, using the provided <Image> information and <Image> OCR text.
            4. Answer in vietnamese language, if not in vietnamese language please translate to vietnamese.
            5. Have full of sentence answer: S + V + adj + ...
            6. Do not include 'Assistant:' in your response
            7. Avoid response in chinese language

        Below is the description of the user's input image and the text in the input image:
        Context: {context}
        OCR text: {ocr_text}

        Answer the question given the image description and text in the image.
        """

        messages = [
            {"role": "system",
            "content": PROMPT_TEMPLATE.format(context=info_img, ocr_text=ocr_text)},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate response
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Response")
        print(response)
        return response