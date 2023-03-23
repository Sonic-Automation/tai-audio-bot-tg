import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import replicate
import openai
from dotenv import load_dotenv
import telebot
import requests
from pydub import AudioSegment
from celery import Celery
import speech_recognition as sr

load_dotenv()

openai.api_key = os.getenv('OPEN_AI_KEY')

app = Celery('chatbot', broker=os.getenv('CELERY_BROKER_URL'))

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

model = replicate.models.get("cjwbw/anything-v3-better-vae")
version = model.versions.get("09a5805203f4c12da649ec1923bb7729517ca25fcac790e640eaa9ed66573b65")

# Store the last 10 conversations for each user
conversations = {}


def image_watermark(img_response):
    """
    :param img_response: image url
    :return: Byte image
    """
    img = Image.open(BytesIO(img_response.content))

    # Add the watermark to the image
    draw = ImageDraw.Draw(img)
    watermark_text = "MuratiAI.com beta"
    font = ImageFont.truetype("anime.ttf", 20)
    # text_size = draw.textsize(watermark_text, font=font)
    # Positioning Text
    x = 6
    y = 6
    # Add a shadow border to the text
    for offset in range(1, 2):
        draw.text((x - offset, y), watermark_text, font=font, fill=(88, 88, 88))
        draw.text((x + offset, y), watermark_text, font=font, fill=(88, 88, 88))
        draw.text((x, y + offset), watermark_text, font=font, fill=(88, 88, 88))
        draw.text((x, y - offset), watermark_text, font=font, fill=(88, 88, 88))
    # Applying text on image murati draw object
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255))

    # Upload the watermarked image to OpenAI and get the URL
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    return img_bytes


@app.task
def generate_image_replicate(prompt):
    inputs = {
        # Input prompt
        'prompt': prompt,

        # The prompt or prompts not to guide the image generation (what you do
        # not want to see in the generation). Ignored when not using guidance.
        'negative_prompt': "lowres, bad anatomy, bad hands, text, error, missing fingers, missing legs, extra digit, "
                           "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
                           "signature, watermark, username, blurry, artist name",

        # Width of output image. Maximum size is 1024x768 or 768x1024 because
        # of memory limits
        'width': 512,

        # Height of output image. Maximum size is 1024x768 or 768x1024 because
        # of memory limits
        'height': 512,

        # Number of images to output
        'num_outputs': 1,

        # Number of denoising steps
        # Range: 1 to 500
        'num_inference_steps': 11,

        # Scale for classifier-free guidance
        # Range: 1 to 20
        'guidance_scale': 7,

        # Choose a scheduler.
        'scheduler': "DPMSolverMultistep",

        # Random seed. Leave blank to randomize the seed
        # 'seed': ...,
    }

    output = version.predict(**inputs)
    return output[0]


@app.task
def generate_response_chat(message_list):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                     {"role": "system",
                      "content": "You are an AI named murati and you are in a conversation with a human. You can answer"
                                 "questions, provide information, and help with a wide variety of tasks."},
                     {"role": "user", "content": "Who are you?"},
                     {"role": "assistant",
                      "content": "I am the murati powered by ChatGpt.Contact me murati@muratiai.com"},
                 ] + message_list
    )

    return response["choices"][0]["message"]["content"].strip()


@bot.message_handler(commands=["start", "help"])
def start(message):
    if message.text.startswith("/help"):
        bot.reply_to(message, "/image to generate image animation\n/create generate image\n/clear - Clears old "
                              "conversations\nsend text to get replay\nsend voice to do voice"
                              "conversation")
    else:
        bot.reply_to(message, "Just start chatting to the AI or enter /help for other commands")


# Define a function to handle voice messages
@bot.message_handler(content_types=["voice"])
def handle_voice(message):
    user_id = message.chat.id
    sender_username = message.from_user.username
    # Download the voice message file from Telegram servers

    # Send processing
    processing_replay = f"processing command by @{sender_username}"
    bot.reply_to(message, processing_replay)

    file_info = bot.get_file(message.voice.file_id)
    file = requests.get("https://api.telegram.org/file/bot{0}/{1}".format(
        TELEGRAM_BOT_TOKEN, file_info.file_path))

    # Save the file to disk
    with open("voice_message.ogg", "wb") as f:
        f.write(file.content)

    # Use pydub to read in the audio file and convert it to WAV format
    sound = AudioSegment.from_file("voice_message.ogg", format="ogg")
    sound.export("voice_message.wav", format="wav")

    # Use SpeechRecognition to transcribe the voice message
    r = sr.Recognizer()
    with sr.AudioFile("voice_message.wav") as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)

    text = text.lower()

    task = generate_image_replicate.apply_async(args=[text])
    image_url = task.get()

    # Define the caption and entities for the image
    space_markup = '                                                                                  '
    image_footer = '[Join muratiAI](https://t.me/muratiAI) | [Website](https://muratiai.com) | [Buy](' \
                   'https://muratiai.com)'
    caption = f"Image generated by @{sender_username} using **[murati](https://muratiai.com)" + "\n" + text + \
              "\n" + image_footer

    if image_url is not None:
        img_response = requests.get(image_url)
        img_bytes = image_watermark(img_response)

        bot.send_photo(chat_id=message.chat.id, photo=img_bytes, reply_to_message_id=message.message_id,
                       caption=caption, parse_mode='Markdown')
    else:
        bot.reply_to(message, "Could not generate image, try again later.")


# Generate response

# Delete the temporary files
os.remove("voice_message.ogg")
os.remove("voice_message.wav")


@bot.message_handler(func=lambda message: True)
def echo_message(message):
    user_id = message.chat.id


if __name__ == "__main__":
    bot.polling()
