import pytesseract
import openai

from pdf2image import convert_from_path




images = convert_from_path("DisclosureSheet.pdf")  # full path if needed

# OCR on the first page
text = pytesseract.image_to_string(images[0])

full_text = ""
for page in images:
    full_text += pytesseract.image_to_string(page) + "\n"

# Azure OpenAI settings
import openai

openai.api_type = "azure"
openai.api_key = "YOUR_AZURE_KEY"
openai.api_base = "YOUR_AZURE_ENDPOINT"
openai.api_version = "2023-12-01-preview"  # or your version

response = openai.ChatCompletion.create(
    engine="YOUR_DEPLOYED_MODEL_NAME",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Extract the key points from this document:\n{full_text}"}
    ]
)

print(response['choices'][0]['message']['content'])
