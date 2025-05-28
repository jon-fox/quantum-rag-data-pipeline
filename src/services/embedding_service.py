\
import os
import logging
from openai import OpenAI
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    A service to generate text embeddings using OpenAI models.
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """
        Initializes the EmbeddingService.

        Args:
            api_key: The OpenAI API key.
            model_name: The name of the OpenAI embedding model to use.
                        Defaults to "text-embedding-3-small".
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for EmbeddingService.")

        self.model_name = model_name
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

        # Determine embedding dimension based on the model name, as per user's logic
        if self.model_name == "text-embedding-3-small":
            self._embedding_dim = 1536
        else:  # For any other model, user's logic implies 3072
            self._embedding_dim = 3072
            # Add a warning if it's not a known model that typically outputs this dimension or supports the 'dimensions' param
            if self.model_name != "text-embedding-3-large":
                logger.warning(
                    f"Model '{self.model_name}' is not 'text-embedding-3-small'. "
                    f"Setting target embedding dimension to {self._embedding_dim}. "
                    f"Ensure this model supports this dimension natively or via the 'dimensions' API parameter."
                )
        
        logger.info(f"EmbeddingService initialized with model: {self.model_name}, target dimension: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Returns the configured embedding dimension for the model."""
        return self._embedding_dim

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
            Returns a zero vector of the configured dimension on error.
        """
        if not text or not isinstance(text, str):
            logger.error("Input text must be a non-empty string.")
            return [0.0] * self._embedding_dim

        try:
            # OpenAI recommends replacing newlines with a space for best results.
            processed_text = text.replace("\\n", " ")
            
            create_params = {"input": [processed_text], "model": self.model_name}
            # The 'dimensions' parameter is supported by text-embedding-3 models
            if self.model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
                create_params["dimensions"] = self._embedding_dim
            
            response = self.client.embeddings.create(**create_params)
            embedding = response.data[0].embedding

            if len(embedding) != self._embedding_dim and self.model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
                 logger.warning(f"Returned embedding dimension {len(embedding)} for model {self.model_name} "
                                 f"does not match target dimension {self._embedding_dim} when 'dimensions' parameter was used. "
                                 "This could indicate an API or model version inconsistency.")
            elif len(embedding) != self._embedding_dim:
                 logger.info(f"Returned embedding dimension {len(embedding)} for model {self.model_name}. "
                                f"Expected based on configuration: {self._embedding_dim}.")


            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: '{text[:50]}...': {e}", exc_info=True)
            return [0.0] * self._embedding_dim # Return zero vector on error

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings (each a list of floats).
            Returns a list of zero vectors on error for the batch.
        """
        if not texts or not all(isinstance(t, str) and t for t in texts):
            logger.error("Input texts must be a list of non-empty strings.")
            return [[0.0] * self._embedding_dim for _ in range(len(texts or []))]

        try:
            processed_texts = [text.replace("\\n", " ") for text in texts]
            
            create_params = {"input": processed_texts, "model": self.model_name}
            if self.model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
                create_params["dimensions"] = self._embedding_dim

            response = self.client.embeddings.create(**create_params)
            embeddings = [data.embedding for data in response.data]
            
            # Optional: Add dimension check for each embedding if necessary
            # for i, emb in enumerate(embeddings):
            #     if len(emb) != self._embedding_dim:
            #         logger.warning(...)

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
            return [[0.0] * self._embedding_dim for _ in texts]
