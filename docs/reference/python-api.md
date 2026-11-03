# Python API

Most projects only need YAML and the CLI. These are the classes to use when
embedding ragkit in your own Python code or writing plugins. Generated from
docstrings.

## Core types

::: ragkit.core.types
    options:
      members: [RawDocument, ParsedContent, Document, Chunk, Query, ScoredChunk, Modality]
      show_root_heading: false

## Plugin interfaces

::: ragkit.core.interfaces
    options:
      show_root_heading: false

::: ragkit.core.registry.Registry

::: ragkit.core.identity
    options:
      members: [Identity, get_identity, acting_as]

## Search

::: ragkit.retrieval.knowledge
    options:
      members: [KnowledgeBase, format_results, record_retrievals]
      show_root_heading: false

::: ragkit.retrieval.stores.base.Store

## Ingestion

::: ragkit.ingest.pipeline
    options:
      members: [IngestPipeline, IngestReport, ingest]
      show_root_heading: false

::: ragkit.ingest.jobs
    options:
      members: [Job, JobQueue, create_queue]
      show_root_heading: false

::: ragkit.ingest.worker.Worker

## Agents

::: ragkit.agents.runtime
    options:
      members: [AgentRuntime, Agent, AgentResult, PendingAction, ApprovalDecision]
      show_root_heading: false

::: ragkit.agents.templates
    options:
      members: [add_agent, list_templates]
      show_root_heading: false

## Evals

::: ragkit.evals.runner
    options:
      members: [evaluate_agent, evaluate_search, check_gates, EvalReport]
      show_root_heading: false

::: ragkit.evals.judge.Judge

## Server and tracing

::: ragkit.server.app.create_app

::: ragkit.observe.configure_tracing
