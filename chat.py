from openai import OpenAI

from secrets import api_key

client = OpenAI(api_key=api_key)

# Go straight to your elected leaders. Although many are already convinced, constantly bring it to their attention. Maybe they can help.

messages = [{"role": "system", "content": "DO NOT REPEAT WHAT THE USER HAS SAID. ANSWER THE USER'S QUESTIONS."}]
# messages = []

while True:
    print("> ", end="")
    question = input()

    if question == "quit":
        break

    messages.append({"role": "user", "content": question})

    # print(messages)

    completion = client.chat.completions.create(
      model="ft:gpt-3.5-turbo-0613:personal::8vySDqIl", #
      messages=messages
    )

    response = completion.choices[0].message.content

    print(response)

    messages.append({"role": "assistant", "content": response})


