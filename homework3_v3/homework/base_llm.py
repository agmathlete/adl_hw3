from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        Generate a response for a single prompt.
        """
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate the response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Using greedy decoding for single generation
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of generate method that handles multiple prompts efficiently.
        """
        # Set tokenizer to pad on the left for generation
        self.tokenizer.padding_side = "left"
        
        # Tokenize all prompts with padding
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate responses
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=temperature > 0,
            temperature=temperature,
            num_return_sequences=num_return_sequences or 1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated tokens
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],  # Only decode the generated tokens
            skip_special_tokens=True
        )
        
        # Reshape the output if num_return_sequences is specified
        if num_return_sequences is not None:
            # Split the flat list into a list of lists
            return [generated_texts[i:i + num_return_sequences] for i in range(0, len(generated_texts), num_return_sequences)]
        
        return generated_texts

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
