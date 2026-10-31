# API limits

The REST API allows 600 requests per minute per token.
Requests over the limit receive HTTP 429 with a Retry-After header.
