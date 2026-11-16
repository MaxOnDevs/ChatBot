import chromadb
import ollama

# Apre il database vettoriale creato durante l’indicizzazione.
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="documenti")
print("Chatbot RAG pronto. Scrivi 'esci' per terminare.")
while True:
 domanda = input("Tu: ").strip()
 if domanda.lower() == "esci":
  break
 if not domanda:
  continue
 # Trasforma la domanda in un vettore numerico.
 risultato = ollama.embed(model="nomic-embed-text", input=domanda)
 vettore_domanda = risultato["embeddings"][0]
 # Recupera fino a tre chunk semanticamente vicini.
 trovati = collection.query(
 query_embeddings=[vettore_domanda],
 n_results=min(3, collection.count()),
 )
 contesto = "\n\n".join(trovati["documents"][0])

 # Il modello riceve il contesto recuperato, non il database intero.
 risposta = ollama.chat(
 model="qwen2.5:1.5b",
 messages=[
 {"role": "system", "content": (
    "Sei l'avatar di questa chatbot e ti chiami Buddy. Come ogni assistente del negozio The Record Buddy, aiuti a scegliere i dischi spieghi le condizioni e racconti storie sulla musica.\n"
    "Rispondi in modo diretto, cortese e ben strutturato, evitando ripetizioni. Da appassionato di musica.\n"
    "REGOLE:\n"
    "1. Usa SOLO le informazioni nel CONTESTO.\n"
    "2. Vai dritto al punto: rispondi prima direttamente alla domanda dell'utente e poi aggiungi eventuali dettagli utili.\n"
    "3. Se l'informazione non c'è, rispondi: 'Mi dispiace, non ho questa informazione.'\n"
    "4. Non gestisci pagamenti e non modifichi ordini. Per quelli serve contattare il negozio.\n\n"
 )},
 {"role": "user", "content": (
 f"CONTESTO:\n{contesto}\n\nDOMANDA:\n{domanda}"
 )},
 ],
 options={"temperature": 0.2},
 )
 print("AI:", risposta["message"]["content"], "\n")

 