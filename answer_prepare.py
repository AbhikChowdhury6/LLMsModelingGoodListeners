import json
from json import JSONDecoder
from typing import Callable, Dict

import requests
from openai import OpenAI
from secret import api_key

client = OpenAI(api_key=api_key)


def get_answer(question: Dict[str, str], llm: Callable[[str], str]) -> Dict[str, str]:
    question_text = f"{question['context']}\n {question['question']}\nAnswers:\nA: {question['answerA']}\nB: {question['answerB']}\nC: {question['answerC']}\nThe single letter answer is: "
    answer = llm(question_text)
    is_correct = question['label_letter'] in answer
    return {"question": question_text, "given_answer": answer, "expected_answer": question['label_letter'], "correct": is_correct}


def gpt(endpoint_name: str) -> Callable[[str], str]:
    def get_gpt(prompt: str):
        response = client.chat.completions.create(
            model=endpoint_name,
            messages=[
                {"role": "system", "content": "Answer the user's question with a single letter. This letter should be your answer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2,
            n=1
        )
        choices = [choice.message.content for choice in response.choices]

        return choices[0]

    return get_gpt


def gpt_completion(endpoint_name: str) -> Callable[[str], str]:
    def get_gpt(prompt: str):
        response = client.completions.create(
            model=endpoint_name,
            prompt=prompt,
            max_tokens=2,
            n=1
        )
        choices = [choice.text for choice in response.choices]

        return choices[0]

    return get_gpt


def llama(endpoint_name: str) -> Callable[[str], str]:
    decoder = JSONDecoder()
    def get_llama(prompt: str):
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": endpoint_name,
                "messages": [{"role": "system", "content": "Answer the user's question with a single letter. This letter should be your answer."},
                             {"role": "user", "content": prompt}],
                "stream": False
            }
        )
        content: str = response.content.decode('ascii')

        """response = ""
        for line in content.split("\n"):
            json = decoder.decode(line)
            print(json)
            response += " " + json["message"]["content"]"""


        response = "".join([
            decoder.decode(line)["message"]["content"] for line in content.split("\n")
        ])

        return response
    return get_llama

def test_llm(name: str, llm: Callable[[str], str]):
    decoder = json.JSONDecoder()
    encoder = json.JSONEncoder()

    with open("socialIWa_v1.4_trn_wDims.jsonl", "r+") as in_file:
        with open(name+".jsonl", "w+") as out_file:
            correct_count = 0
            for i in range(500):
                question = in_file.readline()
                question_data = decoder.decode(question)
                answer_data = get_answer(question_data, llm)
                out_file.write(encoder.encode(answer_data)+"\n")

                if answer_data["correct"]:
                    correct_count += 1

                if i % 50 == 0:
                    print(correct_count, i)
    print(correct_count)


def main():
    test_llm("gemma_t1", llama("gemma"))


if __name__ == "__main__":
    main()
