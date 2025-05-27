provider "aws" {
  region = var.region
}

provider "random" {}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
    }
    random = {
      source  = "hashicorp/random"
    }
  }
  
  backend "s3" {
    key    = "terraform/quantum_rag/terraform.tfstate"  # Path inside the bucket to store the state
    region = "us-east-1"  # AWS region
  }
}

resource "aws_dynamodb_table" "quantum_embeddings" {
  name           = "quantum_embeddings"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "document_id"

  # Primary key attribute
  attribute {
    name = "document_id"
    type = "S"
  }
  
  # Document type attribute for GSI (energy, forecast, analysis)
  attribute {
    name = "doc_type"
    type = "S"
  }
  
  # Last updated timestamp for freshness tracking
  attribute {
    name = "last_updated"
    type = "S"
  }

  # Document type GSI for querying by type (energy data, forecast, analysis)
  global_secondary_index {
    name               = "DocTypeIndex"
    hash_key           = "doc_type"
    projection_type    = "INCLUDE"
    non_key_attributes = ["title", "source", "vector_id", "embedding_type", "summary"]
  }
  
  # Date-based GSI for finding recent documents
  global_secondary_index {
    name               = "DateIndex"
    hash_key           = "last_updated"
    projection_type    = "INCLUDE"
    non_key_attributes = ["document_id", "title", "source", "doc_type", "vector_id"]
  }
  
  # TTL for document expiration
  ttl {
    attribute_name = "expiry_time"
    enabled        = true
  }
  
  # Enable point-in-time recovery for data protection
  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name        = "quantum_embeddings"
    Environment = var.environment
    Project     = "quantum-rag"
    ManagedBy   = "terraform"
    Purpose     = "semantic-reranking"
    CreatedDate = "2025-05-17"
  }
}

# Output the DynamoDB table name
output "dynamodb_table_name" {
  value       = aws_dynamodb_table.quantum_embeddings.name
  description = "Name of the DynamoDB table for quantum embeddings"
}

# Output the DynamoDB table ARN
output "dynamodb_table_arn" {
  value       = aws_dynamodb_table.quantum_embeddings.arn
  description = "ARN of the DynamoDB table for quantum embeddings"
}
