variable "region" {
  description = "The AWS region to deploy the infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (e.g., dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "vpc_id" {
  description = "The ID of the VPC where the RDS instance will be deployed"
  type        = string
  default     = "vpc-0ae37bc4e75d574ad"
}

variable "database_subnet_ids" {
  description = "List of subnet IDs for the RDS subnet group"
  type        = list(string)
  default     = ["subnet-0d35ca21bbec62faa", "subnet-00751115f579e639c"]
}

variable "quantum_simulator_type" {
  description = "Type of quantum simulator to use (e.g., aer_simulator, statevector_simulator)"
  type        = string
  default     = "statevector_simulator"
}

variable "embedding_dimension" {
  description = "Dimension for vector embeddings in the database"
  type        = number
  default     = 1536  # Default for OpenAI embeddings
}

variable "enable_real_quantum_hardware" {
  description = "Whether to allow connection to real quantum hardware"
  type        = bool
  default     = false
}