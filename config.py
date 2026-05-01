"""Central configuration for the local Wikipedia RAG system."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VECTOR_DIR = ROOT / "vector_store"
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

# Ollama local server
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"   # Better retrieval quality than nomic-embed-text on biographies.
LLM_MODEL = "llama3.2:3b"          # Alternatives: "phi3", "mistral"

# Chunking
CHUNK_SIZE = 1200                   # characters per chunk (roughly ~250 tokens)
CHUNK_OVERLAP = 150                 # overlap between chunks for context continuity

# Embedding parallelism (Ollama handles concurrent requests fine).
EMBED_WORKERS = 4

# Retrieval
TOP_K = 4                           # chunks per entity-type returned to the LLM

# Entities required by the homework spec (and a few extras to exceed the 20+20 minimum).
PEOPLE = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare",
    "Ada Lovelace", "Nikola Tesla", "Lionel Messi", "Cristiano Ronaldo",
    "Taylor Swift", "Frida Kahlo", "Isaac Newton", "Charles Darwin",
    "Mahatma Gandhi", "Stephen Hawking", "Mustafa Kemal Ataturk",
    "Mozart", "Vincent van Gogh", "Pablo Picasso", "Steve Jobs",
    "Elon Musk", "Michael Jackson", "Beyonce",
]

PLACES = [
    "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon",
    "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty",
    "Giza pyramid complex", "Mount Everest", "Stonehenge", "Petra",
    "Christ the Redeemer (statue)", "Angkor Wat", "Acropolis of Athens",
    "Sagrada Familia", "Sydney Opera House", "Niagara Falls",
    "Mount Fuji", "Burj Khalifa", "Topkapi Palace", "Galata Tower",
]
