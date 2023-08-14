# Chatbot API Powered by Langchain ,ElevenLabs ,OpenAI

This project aims to create an API that allows users to interact with a chatbot capable of answering various queries based on user-provided data. The chatbot accepts both text and voice inputs from users and responds with text and synthesized voice output.

## Features

- Accepts user queries in text and voice formats.
- Converts user voice queries to text using the Whisper ASR system.
- Utilizes the GPT-3.5 model (powered by OpenAI) to generate textual responses to user queries.
- Employs the ElevenLabs API for generating high-quality voice responses.
- Allows users to provide custom data for the chatbot to answer specific queries.
- Designed for businesses and services to provide automated assistance for common inquiries.

## Technologies Used

- Python
- FastAPI
- GPT-3.5 model (powered by OpenAI)
- Whisper ASR system (for voice-to-text conversion)
- ElevenLabs API (for text-to-voice synthesis)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ahsan-20/Enchant-chatbot-API.git
cd your-chatbot-api
```

2. Install the required dependencies:
 ```bash
 pip install -r requirements.txt
  ```
3. Create a .env file in the same directory and add your OpenAI API key:
 ```bash
OPENAI_API_KEY=your_openai_api_key_here
  ```
## Usage
Start the FastAPI server:
   ```bash
    uvicorn main:app --reload
   ```

## Contributing
Contributions are welcome! If you find any issues or want to add new features, feel free to submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
