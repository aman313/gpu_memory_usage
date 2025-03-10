import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def run_inference_test():
    # Select the best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend - this will be slow for inference")

    # Load a billion parameter transformer model
    model_name = "gpt2-xl"  # GPT-2 XL has 1.5 billion parameters
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Move the model to selected device
    model.to(device)

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded into {device} device. Press 'start' to begin inference or 'exit' to quit.")

    # Keep the script running until the user exits using the keyboard
    try:
        while True:
            user_input = input("Enter 'start' to run inference or 'exit' to quit: ").strip().lower()
            if user_input == 'exit':
                print("Exiting...")
                break
            elif user_input == 'start':
                batch_size = int(input("Enter batch size (default is 1): ").strip() or 1)
                
                # Run inference 20 times
                for i in range(20):
                    input_text = "Hello, how are you?"
                    inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=True).to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Print iteration number and output logits
                    print(f"Iteration {i+1}/20:")
                    print(outputs.logits)
                    
                    # Small delay between iterations
                    time.sleep(0.5)
            else:
                print("Invalid input. Please enter 'start' or 'exit'.")
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    run_inference_test()
