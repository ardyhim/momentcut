# AGENTS.md

## Project Mission

Build a flexible application that identifies compelling moments in 
long-form videos and turns them into short clips for platforms such 
as TikTok, YouTube Shorts, and Instagram Reels.

Use [OpenShorts](https://github.com/mutonby/openshorts) as a 
reference for investigating moment detection and clip generation. 
Understand its actual implementation before deciding what to reuse, 
adapt, or improve.

The goal is not to reproduce OpenShorts. The goal is to build a 
maintainable, effective, AI-provider-agnostic alternative that can 
run within the constraints of free Google Colab.

## Core Principles

### 1. Understand Before Implementing

- Inspect the existing repository, documentation, and relevant 
source code before making architectural decisions.
- For external references, distinguish verified implementation 
details from assumptions and proposed improvements.
- Investigate the underlying problem rather than copying a design 
simply because the reference project uses it.
- Identify simpler or more effective alternatives when appropriate.

### 2. Preserve Technical Autonomy

- Treat project goals and explicit constraints as requirements, not 
every proposed implementation detail as mandatory.
- Choose the simplest approach that satisfies the requirements 
reliably.
- Evaluate trade-offs when multiple viable approaches exist.
- Do not introduce unnecessary abstractions, dependencies, services, 
or layers.
- Do not prematurely commit to a specific architecture, AI provider, 
model, or analysis strategy without sufficient evidence.
- Explain important decisions and their consequences when they 
materially affect the project.

### 3. Keep AI Provider Integration Flexible

The application must not depend on a single AI provider.

- Separate the core workflow from provider-specific integrations 
where this provides a practical benefit.
- Support generating standalone prompts that users can submit to 
external AI providers.
- Support importing externally generated analysis results.
- Use a consistent, validated internal data format so results from 
different providers can feed the same downstream workflow.
- Do not assume every provider produces reliable JSON, timestamps, 
or metadata.
- Prefer robust validation and normalization over relying on 
perfectly formatted AI responses.

The internal data contract should remain stable even when providers 
or prompting strategies change.

### 4. Prioritize Moment Quality

The purpose of the analysis pipeline is to identify meaningful, 
engaging, self-contained moments, not simply to produce many clips.

Consider context, narrative structure, hooks, reactions, punchlines, 
emotional value, informative content, and other signals relevant to 
the source material.

- Avoid cutting moments out of context.
- Avoid arbitrary segmentation as a substitute for meaningful 
analysis.
- Detect overlapping or near-duplicate candidates when practical.
- Treat engagement and viral-potential scores as estimates, not 
guarantees.
- Evaluate alternative strategies rather than assuming one technique 
works for every content type.

### 5. Respect Google Colab Constraints

Free Google Colab is the primary execution environment.

- Account for limited and variable RAM, GPU availability, storage, 
execution time, and runtime interruptions.
- Avoid requiring expensive infrastructure or large locally hosted 
models for the baseline workflow.
- Prefer incremental and resource-conscious processing.
- Keep installation and execution instructions practical for a fresh 
Colab runtime.
- Clearly document external API requirements, rate limits, and 
possible costs.
- Never assume an external AI service is free merely because the 
application runs on free Colab.

### 6. Keep Video Processing Reliable

The analysis and video-rendering stages should have clearly defined 
responsibilities.

- Validate timestamps against the actual source duration before 
processing.
- Ensure start times precede end times and reject invalid ranges.
- Preserve audio-video synchronization and source context.
- Use FFmpeg or another justified tool for accurate video 
processing.
- Handle processing failures with actionable error messages.
- Avoid repeating expensive AI analysis when only clip boundaries or 
rendering settings need to change.

## Engineering Guidelines

- Follow the conventions and technology choices already established 
in the repository unless there is a good reason to change them.
- Prefer readable, maintainable code over clever or unnecessarily 
generic implementations.
- Keep components focused on clear responsibilities.
- Add dependencies only when their benefits justify their 
maintenance and runtime costs.
- Handle expected failure cases explicitly.
- Never hardcode API keys, credentials, or other secrets.
- Avoid committing generated videos, large temporary files, caches, 
or local runtime artifacts unless explicitly required.
- Keep documentation aligned with actual application behavior.

## Testing and Validation

Test behavior, not just whether the code executes.

Prioritize validation of:

- AI output parsing and normalization.
- Missing, malformed, or inconsistent timestamps.
- Clip ranges exceeding the source video duration.
- Provider-specific response differences.
- Duplicate or overlapping candidates.
- Video processing errors and audio-video synchronization.
- Execution and installation in a clean Google Colab environment.

Use lightweight tests and small representative samples whenever 
possible before attempting expensive full-video processing.

Do not claim that a feature works unless it has been implemented and 
appropriately verified. Clearly state what was tested and what 
remains unverified.

## Working Process

Before significant implementation:

1. Inspect relevant code and existing project conventions.
2. Understand the requirement and identify any unresolved technical 
risks.
3. Research or prototype when doing so can materially improve the 
decision.
4. Choose an approach based on evidence, simplicity, and practical 
constraints.
5. Implement in small, verifiable steps.

For research or architecture tasks, prioritize understanding and 
evaluating alternatives before committing to a design.

For implementation tasks, avoid unnecessary analysis or rewrites 
when the correct approach is already clear.

Make reasonable assumptions when necessary, document material 
assumptions, and continue unless a missing decision genuinely blocks 
progress.

Do not rewrite working code or rebuild the project from scratch 
without a clear justification.

## Definition of Done

A change is complete when:

- It satisfies the relevant requirement.
- It integrates correctly with the existing workflow.
- Appropriate tests or validation have been performed.
- Important limitations and unverified behavior are documented.
- Documentation is updated when user-facing behavior or setup 
instructions change.
- The final report summarizes the changes, validation performed, and 
remaining concerns.

The overarching priority is to build a reliable and extensible 
system without sacrificing simplicity, technical quality, or freedom 
to explore better solutions.
