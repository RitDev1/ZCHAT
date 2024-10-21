
Para rodar, é necessário ter o Ollama baixado. 

Baixe o Ollama no site https://ollama.com/

Após baixar, abra o cmd e digite ollama serve

Após isso, digite no cmd ollama pull llama3.1 , ollama pull mistral e ollama pull phi3

Para rodar o aplicado, abra o cmd, digite cd >diretório do arquivo.py< para rodar o cmd no diretório.
Digite, então pip install -r requirements.txt 
Após isso, digite no cmd streamlit run zchat.py para abrir a aplicação web

Coloque na aplicação web os diretórios de entrada e de saída. 

Recomenda-se instalar o driver CUDA para rodar a aplicação web na GPU, caso sua GPU seja compatível com o driver. 