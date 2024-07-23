from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
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
        # QWEN 2
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            quantization_config=self.quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
        
        # LLAVA NEXT
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config
        )
    
    def perform_ocr(self, image):
        result = self.reader.readtext(image)
        ocr_text = "\n".join([res[1] for res in result])
        return "Info of text in image: " + ocr_text
    
    def generate_response(self, image, user_prompt):
        # OCR Processing
        ocr_text = self.perform_ocr(image)
        print("ocr")
        print(ocr_text)
        print("-"*10)
        
        # Prepare conversation and template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text"},
                    {"type": "image"},
                ],
            },
        ]
        chat_template = """[INST] <image>\nDescribe about this image? 
                                            1. Describe about Surroundings Scenes
                                            2. Describe about Key Objects and Activities
                                            3. can include text extraction if possible
                                [/INST]"""
        prompt = self.processor.apply_chat_template(conversation, chat_template=chat_template, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
        
        # Generate output from LLAVA model
        output = self.llava_model.generate(**inputs, max_new_tokens=200)
        info_img = self.processor.decode(output[0], skip_special_tokens=True)
        print("llava")
        print(info_img)
        print("-"*10)
        # Prepare the full prompt for Qwen model
        messages = [
            {"role": "system", "content": """System: You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.
                            Some rules to follow:
                                1. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
                                2. You must focus on OCR task.
                                3. Answer question from <User> with clarification and high precision, using the provided <Image>.
                                4. Answer in vietnamese language, if not in vietnamese language please translate to vietnamese.
                                5. Have full of sentence answer: S + V + adj + ...
                                6. Do not include 'Assistant:' in your response
                                7. Avoid response chinese language
                                """},
            {"role": "user", "content": "\n Please Answer in vietnamese language." + user_prompt + info_img + ocr_text}
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