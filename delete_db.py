import chromadb

db_path = "./vector_db"
client = chromadb.PersistentClient(path=db_path)

# collections = client.list_collections()
#
# print("现有Collections:")
# for collection in collections:
#     print(f"- {collection.name}: {collection.metadata}")

client.delete_collection(name="qa_embeddings_bge")
print("Collection已删除")
