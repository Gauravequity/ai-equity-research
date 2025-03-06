import openai

openai_client = openai.OpenAI(api_key="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")  # Replace with your actual OpenAI API key

try:
    models = openai_client.models.list()
    model_list = [model.id for model in models.data]  # Extract available models
    print("✅ Available Models:", model_list)

    # Check if GPT-4 Turbo is available
    if "gpt-4-turbo" in model_list:
        print("🎉 Your API key has access to GPT-4 Turbo!")
    else:
        print("⚠️ Your API key does NOT have access to GPT-4 Turbo.")
        print("Check your OpenAI account to upgrade or enable API access.")

except openai.OpenAIError as e:
    print(f"❌ API Error: {e}")
