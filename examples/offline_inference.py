from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    "Tell me something about Canada",
    # "How are you today?",
    # "What is your business?",
    # "I don't know what to ask"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, seed=42, max_tokens=30, use_beam_search=True, best_of=4, n=4)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    for output in output.outputs:
        generated_text = output.text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

###########################
# from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# set_seed(42)

# model_id = "facebook/opt-125m"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# for prompt in prompts:
#     inputs = tokenizer.encode(prompt, return_tensors='pt')
#     output = model.generate(inputs, max_new_tokens=30, do_sample=False)
#     text = tokenizer.batch_decode(output)
#     print(text)
