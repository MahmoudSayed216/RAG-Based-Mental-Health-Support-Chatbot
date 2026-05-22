# from qdrant_client import QdrantClient

# client = QdrantClient(path="./qdrant_db")

# info = client.get_collection("mental_health")
# print(f"Total vectors: {info.points_count}")
# print(f"Vector size:   {info.config.params.vectors.size}")
# print(f"Distance:      {info.config.params.vectors.distance}")


# results = client.scroll(
#     collection_name="mental_health",
#     limit=5,
#     with_payload=True,
#     with_vectors=True
# )

# print("\n--- Sample records ---\n")
# for point in results[0]:
#     print(f"ID:      {point.id}")
#     print(f"Payload: {point.payload}")
#     print(f"Vector:  {point.vector[:5]}...")  # first 5 values, it's 1024 floats so don't print all
#     print()
