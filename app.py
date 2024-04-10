import requests
from openai import AzureOpenAI
import tiktoken
import pyperclip
import os
import urllib.parse

class AIAssistant:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key="openai-key",
            api_version="2024-02-01",
            azure_endpoint="https://andre-openai3.openai.azure.com/"
        )
        self.api_key = "ai-search-api-key"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    def detect_intent(self, query):
        promptIntent = f"""Você precisa classificar a pergunta do usuário em algumas dessas intenções:  
        - Pergunta e resposta: a resposta para a pergunta pode ser achada fazendo uma busca objetiva  
        - Resumo: a resposta para a pergunta precisa de uma informação de contexto mais amplo e portanto precisa fazer um resumo para poder responder com maior contexto  

        Pergunta do usuário:
        {query}

        Intenção:"""

        responseIntent = self.client.chat.completions.create(
            model="andre-gpt4",
            messages=[
                {"role": "user", "content": promptIntent}
            ],
            max_tokens=100,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False
        )

        responseIntent = responseIntent.choices[0].message.content

        print("Intent: " + responseIntent)

        return "resumo" in responseIntent.lower()

    def search_documents(self, query, folder, is_summarization):
        if is_summarization:
            search_url = "https://andre-search-basic.search.windows.net/indexes/vector-tjrj-doc/docs/search?api-version=2024-03-01-preview"
        else:
            search_url = "https://andre-search-basic.search.windows.net/indexes/vector-tjrj-chunk/docs/search?api-version=2024-03-01-preview"

        if is_summarization:
            payload = {
                "search": "*",
                "filter": f"folder eq '{folder}'"
            }
        else:
            payload = {
                "search": query,
                "queryType": "semantic",
                "semanticConfiguration": "vector-tjrj-semantic-configuration",
                "queryLanguage": "pt-BR",
                "top": "5",
                "select": "chunk,url"
            }

        response = requests.post(search_url, json=payload, headers=self.headers)
        response_json = response.json()

        prompt = ""
        for result in response_json["value"]:
            url = result["url"]
            decoded_url = urllib.parse.unquote(url)
            prompt += "[" + decoded_url.split("/")[-1] + "]\n"
            if is_summarization:
                prompt += result["summary"]
            else:
                prompt += result["chunk"].replace("\n", "")
            prompt += "\n"
        prompt += "\n"

        pyperclip.copy(prompt)

        return prompt

    def get_response(self, query, folder):
        is_summarization = self.detect_intent(query)

        prompt = self.search_documents(query, folder, is_summarization)

        tokens_prompt = self.num_tokens_from_string(prompt)
        print(f"Tokens prompt: {tokens_prompt}")

        prompt_openai = f"""Vocé é um assistante que responde a perguntas sobre processos jurídicos. Instruções:
        - Responda com um linguajar jurídico, como um advogado ou juiz
        - Os detalhes do processo estão abaixo definidos na seção "PROCESSO:". Use essas informações para responder a pergunta. Um processo é composto por vários documentos. Cada documento está definido abaixo no formato [nome-do-documento]. Use o nome do documento como referência para responder a pergunta.
        - Se a pergunta está pedindo sobre alguma lei, você pode utilizar o seu conhecimento para responder sobre essa lei do Brasil, não precisa se limitar somente ao conteúdo do processo.
        ===
        PROCESSO:
        {prompt}
        ===
        PERGUNTA:
        {query}
        ====
        Resposta:
        """

        print("Obtendo resposta...")

        response = self.client.chat.completions.create(
            model="andre-gpt4",
            messages=[
                {"role": "user", "content": prompt_openai}
            ],
            max_tokens=4000,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=False
        )

        response = response.choices[0].message.content

        print("RESPOSTA:")
        print(response)

    @staticmethod
    def num_tokens_from_string(string: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(string))
        return num_tokens


folder = "0813891-54.2023.8.19.0031"
query = """Verifique o período da alegada ausência do serviço e se durou mais de 24 horas."""


assistant = AIAssistant()
assistant.get_response(query, folder)


