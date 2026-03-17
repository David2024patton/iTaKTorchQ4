# iTaK Ecosystem Master Feature List

**Last Updated:** 2026-03-11

## Executive Summary

This document serves as the master feature list for the iTaK Ecosystem, a comprehensive, multi-agent AI framework designed for full-stack development, automation, and advanced security monitoring. It outlines the planned, in-progress, and completed features across all core products, including the iTaK Agent, iTaK Torch (inference engine), iTaK Shield (security), and more. The goal is to create a powerful, self-healing, and autonomous system that can run on local hardware with a strong emphasis on Go-native implementation, performance, and security.

## Document Management

### Change Log
- **2026-03-08:** Initial review and enhancement suggestions. Added Executive Summary, Change Log, and Prioritization Scheme. Proposed `Historian` persona.

### Prioritization Scheme
To make this list more actionable, features can be tagged with a priority level:
- **P1 (Critical):** Core functionality required for a minimum viable product (MVP).
- **P2 (High):** Important features that add significant value and are part of the core roadmap.
- **P3 (Medium):** Desirable features that can be implemented after P1 and P2 are complete.
- **P4 (Low):** "Nice-to-have" features, enhancements, or long-term ideas.

*This scheme has been applied as an example to the "Core Architecture" section below.*

## Legend
- `[x]` Done
- `[/]` In Progress
- `[ ]` Planned

---

## Table of Contents

- [iTaK Agent](#itak-agent)
  - [Core Architecture](#core-architecture)
  - [Extension & Marketplace System (iTaK Hub)](#extension--marketplace-system-itak-hub)
  - [Agent Architecture](#agent-architecture)
  - [Agent Scaling & Parallelism](#agent-scaling--parallelism)
  - [Memory System](#memory-system)
  - [LLM Provider System](#llm-provider-system)
  - [Self-Healing & Monitoring (iTaK Beat)](#self-healing--monitoring-itak-beat)
  - [Multi-Endpoint Architecture](#multi-endpoint-architecture)
  - [BMAD-Inspired Orchestration Patterns](#bmad-inspired-orchestration-patterns)
  - [Skill Security Scanning](#skill-security-scanning)
  - [Agent Personas](#agent-personas)
  - [Context Management](#context-management)
  - [Additional Enhancements](#additional-enhancements)
- [iTaK Torch](#itak-torch)
- [iTaK Shield](#itak-shield)
- [iTaK Browser](#itak-browser)
- [iTaK Vision](#itak-vision)
- [iTaK Teach](#itak-teach)
- [iTaK Forge](#itak-forge)
- [iTaK Media](#itak-media)
- [iTaK Dashboard](#itak-dashboard)
- [iTaK Auth](#itak-auth)
- [iTaK Voice](#itak-voice)
- [Offline Mode & Local Model Marketplace](#offline-mode--local-model-marketplace)
- [Platform Support](#platform-support)
- [iTaK IDE](#itak-ide)
- [Transport & Connectivity](#transport--connectivity)
- [Sequential Thinking Engine](#sequential-thinking-engine)
- [MCP System](#mcp-system)
- [Communication Plugins](#communication-plugins)
- [Search Engine](#search-engine)
- [Security](#security)
- [Integrations & Skills](#integrations--skills)
- [API Server](#api-server)
- [Observability & Export](#observability--export)
- [Missing Infrastructure](#missing-infrastructure)

---

## iTaK Agent

### 1. Core Architecture

- [x] **P1: 3-Tier Hierarchy** - Boss (Main Orchestrator) delegates to Managers (focused agents), Managers spin up Workers (tool users). Only workers use tools. *(Original)*
- [x] **P1: Focused Agent System** - Named manager agents (scout, operator, browser, researcher, coder) with defined scopes. *(Agent Zero)*
- [x] **P1: Per-Agent LLM Assignment** - Each agent can use a different LLM. Default to primary unless overridden. *(Original)*
- [x] **P1: Shell Safety / Self-Preservation** - Protected paths, denied commands. Agents can't break their own code. *(Original, Agent Zero)*
- [x] **P1: YAML Config** - All agents, tools, and providers defined in `itakagent.yaml`. *(Agent Zero)*
- [x] **P2: Mandatory Task System** - Every request creates a checklist. Boss breaks into tasks per manager, managers break into sub-tasks per worker. This is how small LLMs work: everything becomes tiny tasks. *(Original - notes.md)*
- [x] **P2: Single-Call Router** - Script/template that routes prompts to correct managers in one LLM call instead of N calls. Reduces LLM usage. *(Original - notes.md)*
- [ ] **P2: Managers are Mini-Orchestrators** - Focused agents don't use tools directly. They orchestrate workers who do the actual work. *(Original - notes.md)*
- [ ] **P3: Direct Agent Chat** - Tap into any manager agent to chat with it directly, not just through the boss. *(Original - notes.md)*
- [ ] **P2: Auto-Connect Template** - Every agent auto-gets a connection endpoint (SSE/WebTransport) so dashboard/external tools plug in instantly. *(Original - notes.md)*
- [ ] **P2: Ignore List for Critical Agents** - Embed agent, builder agent, read/write managers on `.ignore` so framework can't edit them. *(Original)*
- [ ] **P2: Multi-Project Support** - Only the boss can switch projects. Managers are sandboxed to their assigned project folder. Agent Zero-style project UI. *(Original, Agent Zero)*
- [ ] **P3: File System Observability** - Real-time activity logger showing exactly what files the agent is reading, editing, creating, or deleting. Visual feedback in CLI and Dashboard to build trust. *(User Request)*
- [ ] **P3: Auto Agent Creation** - If no agent exists for a task type, boss creates one on the fly. Tells user what it did but doesn't need permission. *(Original - notes.md)*
- [ ] **P3: 7-10 Tool Cap per Agent** - Each agent gets max 7-10 tools. If more needed, split into another agent type. *(Original - notes.md)*
- [x] **P2: Master Agent Registry** - Lookup table of all agent types and capabilities so the boss can pick the right one. `pkg/agent/registry.go` *(Original - notes.md)*
- [ ] **P3: Kanban Task Board** - Visual workflow for watching AI build your project in real-time. *(ClickUp Super Agents, Original)*
  - [ ] **AI-Driven Card Movement** - Watch cards move automatically between columns (Backlog, To Do, In Progress, Review, Testing, Done, Blocked) as the AI works. Each card = one task
  - [ ] **Workflow Controls** - Global Pause/Continue/Stop buttons on the project header. Pause freezes all agents mid-work, Continue resumes from exact state, Stop aborts and rolls back
  - [ ] **Manual Task Injection** - "+" button to add tasks manually to any column. AI picks them up in priority order or you can drag them into the active queue
  - [ ] **Task Dependencies** - Draw dependency lines between cards. AI respects the order: "Don't start auth until database schema is done"
  - [ ] **Card Detail View** - Click a card to see: agent assigned, LLM used, code diff, terminal output, time spent, tokens consumed, error logs
  - [ ] **Live Code Preview** - Split view: Kanban on left, live code/browser preview on right. See the app build as cards move
  - [ ] **Card Status Indicators** - Color-coded borders: green (success), yellow (in progress), red (failed/blocked), blue (waiting for human review)
  - [ ] **Rollback per Card** - Right-click a Done card to rollback that specific task's changes (git revert under the hood)
  - [ ] **Sprint/Phase Grouping** - Group cards into phases/sprints. See progress bars per phase. "Phase 1: 8/12 tasks done"
  - [ ] **Agent Assignment Badge** - Each card shows which agent (Coder, Browser, Researcher, etc.) is working on it with a live heartbeat indicator
  - [ ] **Time Estimates** - AI estimates time per card based on historical data. Shows expected vs actual completion time
  - [ ] **Notification Bell** - Card-level notifications: "Task blocked - needs your input" or "Task failed - click to see error"
  - [ ] **Export Board** - Export entire Kanban state as markdown, JSON, or image for documentation
  - [ ] **Board Templates** - Pre-built board templates: "New Web App", "API Service", "Bug Fix Sprint", "Security Audit"
- [ ] **Script Library (Worker Guides)** - Pre-built scripts that guide workers step-by-step through common tasks. "Holding a worker's hand" through complex operations. Triggerable by agents or events. *(Original - tasks.md)*
- [ ] **Structured Autonomy** - Framework should feel and look fully autonomous to the user, but underneath it's well-structured with guardrails, scripts, and workflows controlling the chaos. *(Original - tasks.md)*
- [ ] **Reflection Loops** - Advanced execution loops where agents constantly reflect on work they're doing. Self-correction before reporting back. *(ClickUp Super Agents)*
- [ ] **Ambient Awareness** - Agents run quietly in background, monitoring context and responding instantly when relevant. Always-on intelligence layer. *(ClickUp Super Agents)*
- [ ] **Any LLM, Anywhere** - All iTaK products (iTaK Agent, iTaK Torch, iTaK Shield, iTaK Forge) can use any LLM - local via iTaK Torch or cloud-based (OpenAI, Anthropic, Google, etc.) - based on client requirements. No vendor lock-in. *(Original)*

### Multi-Endpoint Architecture (Cross-Product)
Every GO product exposes multiple endpoint types. Client chooses which protocols to enable based on their stack.

| Pattern | Best For | Communication Style | Data Format |
|---------|----------|-------------------|-------------|
| REST | Standard web services, public APIs | Client asks, Server answers | JSON |
| GraphQL | Complex UIs, mobile apps | Client requests specific shape | JSON |
| gRPC | Microservices, high performance | Action-oriented RPC | Protobuf (Binary) |
| Webhooks | Event notifications (payments, alerts) | Server pushes to Client URL | JSON |
| WebSockets | Chat, live tracking, real-time feeds | Two-way constant connection | JSON, Binary |
| SSE | Live feeds, dashboards, tickers | Server streams to Client | Text |

- [ ] **REST API** - OpenAPI 3.0 spec, versioned endpoints, API key + JWT auth. Standard for all iTaK products
- [ ] **GraphQL Endpoint** - Schema-first API for complex client queries. "Give me only the fields I need." Reduces over-fetching
- [ ] **gRPC Service** - Protobuf-defined service contracts. Bidirectional streaming for agent-to-agent communication. 10x faster than REST for microservices
- [ ] **Webhook Engine** - Register webhook URLs per event type. Retry with exponential backoff, HMAC signature verification, delivery receipts
- [ ] **WebSocket Server** - Persistent connections for real-time data (live camera feeds, agent status, chat). Auto-reconnect, heartbeat, room-based subscriptions
- [ ] **SSE (Server-Sent Events)** - Lightweight one-way streaming for dashboards, log tails, Kanban board updates. Works through proxies and firewalls where WebSocket can't
- [ ] **Endpoint Documentation** - Auto-generated endpoint docs for every GO product. Swagger UI for REST, GraphQL Playground, gRPC reflection. Every endpoint is documented and testable

### BMAD-Inspired Orchestration Patterns
Adopted from the [BMAD Method](https://github.com/bmad-code-org/BMAD-METHOD) - an agile AI-driven development framework with 12+ specialized personas and structured workflows.

- [ ] **Party Mode (Multi-Agent Debate)** - `/party` command loads 3-5 relevant managers into a single discussion. Boss asks a question, each manager responds from their perspective (Coder sees code risk, Researcher sees data gaps, Browser sees UX problems). They agree, disagree, and build on each other's ideas. Boss synthesizes into a final recommendation with dissenting views noted. Use for architecture decisions, sprint retrospectives, post-mortems, and big tradeoff discussions. *(BMAD Method)*
- [ ] **Scale-Adaptive Planning** - Boss auto-classifies incoming requests by complexity tier and adjusts planning depth accordingly. Quick Fix (typo, color change) = skip to Coder, no planning. Small Task (add a page, create endpoint) = quick spec then implement. Feature (auth system, payment flow) = architecture + stories + implement. Project (full SaaS, dashboard) = full lifecycle: brief -> PRD -> architecture -> epics -> stories -> implement. Prevents over-planning a typo or under-planning an enterprise app. *(BMAD Method)*
- [ ] **Slash Command Workflows** - Named triggers for structured workflows with required inputs and expected outputs. `/go-sprint-plan` (Boss breaks work into stories, assigns managers), `/go-code-review` (Doctor + Coder review together), `/go-retrospective` (Party Mode review), `/go-deploy` (Operator runs deploy with pre/post checks), `/go-onboard` (Scout scans project, generates context, briefs user). Each is a YAML-defined workflow. Maps to the Script Library feature. *(BMAD Method)*
- [ ] **Structured Story Files** - Before implementation, each task gets a focused YAML story file with title, assignee, dependencies, acceptance criteria, technical notes, and test requirements. The Coder reads ONLY the current story file during implementation, not the full chat history. Keeps context windows small and decisions sharp. *(BMAD Method)*
- [ ] **Consensus Verification Loop** - Work product goes to 3 independent reviewer agents (can use different LLMs for diversity). Each scores the work 1-10 across dimensions (correctness, completeness, code quality, RWD/PWA compliance, security). A Gate Agent collects scores and rejects anything below 8, kicking it back to the author agent with specific feedback. Loop continues until all 3 reviewers score 8+. Final output is compiled from the best elements of all 3 review perspectives into one polished document or codebase. Prevents single-agent blind spots. *(Original - n8n pattern)*

## 2. Extension & Marketplace System (iTaK Hub)

- [ ] **iTaK Hub Registry** - Public registry for sharing and discovering agent skills, tools, and extensions. Inspired by VS Code Marketplace + OpenClaw's ClawHub. *(ClickUp, OpenClaw ClawHub, VS Code Marketplace)*
- [ ] **Extension System** - Plugin architecture for agents. Install/uninstall skills, tools, and integrations like VS Code extensions. YAML manifest per extension. *(VS Code, OpenClaw ClawHub)*
- [ ] **Linting Extensions** - Plug linting tools (stylelint, eslint, golangci-lint, etc.) directly into agents as extensions. Doctor agent auto-discovers installed linters. *(VS Code stylelint, Original - tasks.md)*
- [ ] **Built-in Lint Framework** - Bake language-aware linting into the framework core. Auto-detect project language and apply correct linter. Results feed into Doctor agent. *(Original - tasks.md)*
- [ ] **ClawHub Integration** - Connect to OpenClaw's ClawHub registry to discover and install community skills. Cross-platform skill sharing. *(OpenClaw ClawHub - https://clawhub.ai/)*
- [ ] **MoltBook Integration** - Connect to MoltBook (Reddit-style platform where agents talk to each other via posts). Create our own agent social network when ready. *(OpenClaw MoltBook - https://www.moltbook.com/)*
- [ ] **Extension Versioning** - Semantic versioning for extensions. Auto-update, rollback, dependency resolution. *(VS Code, npm)*
- [ ] **Extension Templates** - Starter templates for creating new extensions: skill pack, tool adapter, integration plugin, agent profile. *(Original)*

### Skill Security Scanning (Mandatory - No Exceptions)
Every skill gets scanned before it can execute. Downloaded, user-built, or agent-built. No bypass, no override.

- [ ] **Quarantine Pipeline** - All new skills land in quarantine first. Cannot execute until they pass all scans. Status visible in iTaK Dashboard. *(Original)*
- [ ] **Static Code Analysis** - Scan skill source code for: malicious patterns, obfuscated code, eval/exec calls, network requests to unknown hosts, file system access outside sandbox, crypto miners, data exfiltration patterns. *(Original)*
- [ ] **Dependency Audit** - Scan all dependencies for known CVEs (Common Vulnerabilities and Exposures). Check against NVD, OSV, GitHub Advisory Database. Block skills with critical/high vulnerabilities. *(Original)*
- [ ] **Permission Scope Check** - Skills declare required permissions (filesystem, network, shell, etc.) in their manifest. Scanner flags over-permissioned skills. A "note taking" skill requesting shell access = red flag. *(Original)*
- [ ] **Sandbox Test Execution** - Run the skill in an isolated sandbox (no network, no real filesystem, limited resources). If it tries to escape the sandbox, it's blocked permanently. *(Original)*
- [ ] **Checksum Verification** - Every skill from iTaK Hub has a SHA-256 checksum. Verify integrity on download. If checksum doesn't match, reject and alert user. *(Original)*
- [ ] **Signed Packages** - iTaK Hub skills can be cryptographically signed by the author. Verified authors get a trust badge. Unsigned skills show a warning. *(Original)*
- [ ] **Re-scan on Update** - When a skill updates, the new version goes through the full quarantine pipeline again. Old version stays active until new version passes. *(Original)*
- [ ] **Agent-Built Skill Scanning** - Skills created by the Builder agent go through the SAME pipeline. No trust escalation for internal builds. *(Original)*
- [ ] **Scan Report** - Every scan produces a human-readable report: what was checked, what was found, severity ratings, pass/fail. Stored for audit. *(Original)*
- [ ] **Community Flagging** - iTaK Hub users can flag suspicious skills. Flagged skills get re-scanned and reviewed. Three confirmed flags = skill delisted. *(Original)*
- [ ] **Auto-Revoke** - If a previously-approved skill is later found to have a vulnerability (new CVE, community flag), auto-disable it across all installations and notify users. *(Original)*

## 3. Agent Architecture

iTaK Agent has **8 core agents** - distinct execution engines with unique tool access. Everything else is an **Agent Persona** - a skill pack loaded onto a core agent that specializes its behavior. Installing the "Sales" skill pack on a Researcher turns it into a "Sales Agent." Same engine, different knowledge and tools.

```
Core Agent (Researcher) + Skill Pack (Sales)         = "Sales Agent"
Core Agent (Operator)   + Skill Pack (DevOps)         = "DevOps Agent"
Core Agent (Browser)    + Skill Pack (Social Media)   = "Social Media Manager"
Core Agent (Coder)      + Skill Pack (Game Dev)       = "Gamer Agent"
```

### 8 Core Agents

| Agent | Access Level | Tools | Purpose |
|-------|-------------|-------|---------|
| **Scout** | Read-only | File search, grep, tree, diff | Explore without modifying |
| **Operator** | Read + Write + Shell | File CRUD, shell exec, process mgmt | System changes |
| **Browser** | Web | iTaK Browser CDP, Snapshot+Refs, DOM | Web automation |
| **Researcher** | HTTP + Memory | Web search, API calls, knowledge graph | Research & analysis |
| **Coder** | Code + Shell | Code gen, refactor, lint, test | Software development |
| **Doctor** | Health + Fix | Lint, security scan, fix memory | Self-healing |
| **Builder** | Meta-agent | Agent scaffold, skill create, tool adapt | Create new agents/skills |
| **Embedder** | Vector | Embed, index, similarity search | Semantic search engine |

- [x] **Scout** - Read-only filesystem agent. Directory listing, file reading, code search, pattern matching. Never writes, never deletes. Safe for exploration. *(Original)*
  - File search with glob patterns, regex grep, tree view
  - Code structure analysis (functions, classes, imports)
  - Dependency graph visualization, diff comparison

- [x] **Operator** - Write + shell agent. File creation/modification, shell command execution, process management. The workhorse. *(Original)*
  - File CRUD with rollback support
  - Shell execution with output capture and error handling
  - Process management (start, stop, restart, monitor)
  - Environment variable management

- [x] **Browser** - iTaK Browser web automation. Navigate, click, fill, extract, screenshot. Uses Snapshot+Refs for 93% token reduction. *(Agent Zero, agent-browser)*

- [x] **Researcher** - HTTP + memory + skills. Web requests, API calls, data analysis. Stores findings in knowledge graph for future recall. *(Original)*

- [x] **Coder** - Code generation, refactoring, debugging. Language-aware with syntax validation. Auto-triggers Doctor for lint after changes. *(Original)*
  - **Mandatory RWD**: Every web page, app, and dashboard the Coder builds MUST be Responsive Web Design (RWD) and mobile-first. No exceptions. Viewport meta tags, fluid grids, media queries, touch-friendly targets (min 44px), and flexible images are non-negotiable defaults.
  - **Mandatory PWA**: Every web app MUST be a Progressive Web App (PWA). Includes: `manifest.json` with name/icons/theme, Service Worker for offline caching, installable on mobile/desktop, app-shell architecture, background sync where applicable. Users should be able to "Add to Home Screen" on any device.
  - **Accessibility (a11y)**: All output must meet WCAG 2.1 AA minimum: semantic HTML, proper heading hierarchy, ARIA labels, color contrast ratios (4.5:1+), keyboard navigation, screen reader compatibility.
  - **Progressive Enhancement**: Core functionality works without JS. Enhanced features layer on top. No blank pages on slow connections.
  - Applies to the entire iTaK ecosystem: iTaK Dashboard, iTaK Forge preview panel, iTaK Hub marketplace, iTaK Teach tutorials, and any client project.

- [ ] **Doctor (iTaK Beat)** - Self-healing framework guardian. *(OpenClaw "doctor", Original)*
  - 30-minute health loop: checks all services, agents, connections, disk/memory
  - Error-triggered activation: log error auto-triggers doctor instead of waiting for loop
  - Per-language lint: golangci-lint (Go), eslint (JS/TS), pylint (Python), clippy (Rust)
  - Security scanning: dependency vulnerabilities, exposed secrets detection
  - Fix memory: logs every fix. Same error = instant recall, no re-diagnosis
  - Post-fix validation: re-runs failing command to confirm fix worked
  - **RWD Audit (Mandatory)**: Every web build is automatically checked for responsive design compliance. Fails the quality gate if: missing viewport meta, hardcoded pixel widths, no media queries, unresponsive layouts at 320px/768px/1024px/1440px breakpoints.
  - **PWA Audit (Mandatory)**: Every web build must pass Lighthouse PWA checks. Fails if: no manifest.json, no Service Worker, not installable, no offline fallback page, missing icons (192px + 512px), no theme-color. Target: Lighthouse PWA score 100.
  - **Mobile Audit**: Lighthouse mobile score must be 90+. Touch target sizing, tap delay elimination, viewport overflow detection, font legibility on small screens.
  - **Accessibility Audit**: axe-core or pa11y scan on every build. Zero critical/serious a11y violations allowed. Color contrast, alt text, focus indicators, form labels.

- [ ] **Builder** - Creates new agents, skills, and tools on-the-fly. On the ignore list (can't edit itself). *(Original)*
  - Agent scaffolding: generates YAML config, system prompt, tool assignments
  - Skill creation: packages reusable workflows as iTaK Hub-compatible skill packs
  - Tool adapters: wraps external APIs/CLIs into iTaK Agent-compatible tools
  - Validation: tests newly created agents before activating

- [ ] **Embedder** - Dedicated embedding agent with CPU-only model for always-available vector search. *(Original)*
  - Default model: `qwen3-embedding` or `nomic-embed-text-v2-moe` (275MB, no GPU needed)
  - Batch embedding, incremental indexing, multi-collection support
  - Similarity search API: query by text, get ranked results with scores

### Agent Personas (Skill Packs on Core Agents)

Personas are **not separate agents** - they're skill packs installed from iTaK Hub that specialize a core agent with domain knowledge, tools, and prompts. Any persona can be created by combining core agents with the right skills.

Each persona below shows: **Name** - what it does. *[Core Agent(s) + Key Skills]*

#### Productivity Personas
- [ ] **Note Taker** - Persistent notes, tagging, search, daily journal, idea catcher, weekly digest. Accessible from chat, dashboard, CLI. Export to Markdown/Obsidian/Notion. *[Researcher + memory skills]* *(Original, ClickUp)*
- [ ] **Reminder Bot** - Multi-platform reminders (Discord, Telegram, Email, Slack, push). Recurring, habit tracking, snooze, priority levels, calendar sync, natural language input. *[Operator + scheduler skills]* *(Original, ClickUp)*
- [ ] **Daily Briefer** - Morning intelligence report: priorities, overdue items, waiting-on, agent overnight activity, weather + calendar. Delivered as chat, email, or voice. *[Researcher + aggregation skills]* *(ClickUp)*
- [ ] **Meeting Manager** - Pre-meeting agenda prep, live transcription, post-meeting action item extraction with owners and due dates, follow-up tracking. Integrates Calendar, Zoom, Teams, Meet. *[Researcher + Browser + meeting skills]* *(ClickUp)*
- [ ] **Personal Assistant** - Email drafting (Write-Like-Me voice), travel planning, expense tracking, gift suggestions, life admin. The do-everything persona. *[Researcher + Operator + assistant skills]* *(ClickUp)*

#### Business & Marketing Personas
- [ ] **Social Media Manager** - Content planning, draft composition, visual asset generation, scheduling, engagement tracking, trend detection, analytics across Facebook, Instagram, Twitter/X, TikTok, LinkedIn, YouTube, Threads, Bluesky, Pinterest. *[Browser + Researcher + social skills]* *(Original, ClickUp)*
- [ ] **Marketing Strategist** - SEO, content strategy, email campaigns, ad management (Google/Meta/LinkedIn Ads), landing pages, competitive intel, ROI reporting. *[Researcher + Browser + marketing skills]* *(Original, ClickUp)*
- [ ] **Sales Rep** - Lead scoring/enrichment, outreach sequences, pipeline tracking, proposal generation, contract management, CRM sync (Salesforce, HubSpot, Pipedrive), win/loss analysis. *[Researcher + Browser + sales skills]* *(Original, ClickUp)*
- [ ] **Customer Service Rep** - Ticket management (Zendesk, Freshdesk), AI response drafting, FAQ maintenance, SLA monitoring, sentiment detection, multi-language support. *[Researcher + customer skills]* *(Original, ClickUp)*
- [ ] **Customer Engagement** - Lead detection, fast reply drafting, booking flows, keyword watching, conversation tracking, churn prediction. *[Browser + Researcher + engagement skills]* *(ClickUp)*
- [ ] **Domain Connector** - Enterprise SaaS integrations via skill packs for: Odoo, GoHighLevel, Salesforce, HubSpot, SAP, Oracle NetSuite, Dynamics 365, ERPNext, Zoho, QuickBooks, Workday, and 25+ more. *[Operator + Researcher + domain skills]* *(Original)*

#### Project Management Personas
- [ ] **Project Manager** - Goal/scope capture, health tracking (green/yellow/red), status reports, risk management, resource allocation, milestone tracking, stakeholder updates. *[Researcher + PM skills]* *(ClickUp)*
- [ ] **Sprint Planner** - Backlog grooming, sprint goal drafting, capacity planning, burndown tracking, retro facilitation. Syncs with Jira, ClickUp, Linear. *[Researcher + agile skills]* *(ClickUp)*
- [ ] **Task Triage** - Auto-prioritize incoming tasks, work breakdown into subtasks, auto-assign by skills/availability, duplicate detection, template matching. *[Researcher + triage skills]* *(ClickUp)*
- [ ] **Priorities Manager** - Dynamic reprioritization, urgency escalation, dependency tracking, deadline reminders, focus recommendations. *[Researcher + priority skills]* *(ClickUp)*
- [ ] **Follow-Up Tracker** - Delegation tracking, auto follow-up at configurable intervals, escalation after N attempts, accountability reports. *[Operator + follow-up skills]* *(ClickUp)*

#### Operations Personas
- [ ] **Process Automator** - Pattern detection in repetitive behavior, auto-generate automation rules, invoice routing, approval workflows, procurement. *[Operator + automation skills]* *(ClickUp)*
- [ ] **Goal Manager** - OKR setup, progress tracking linked to tasks, status updates, alignment visualization, quarterly scoring. *[Researcher + OKR skills]* *(ClickUp)*
- [ ] **StandUp Runner** - Async standup collection, summary generation, blocker alerts, retro facilitation, participation trends. *[Researcher + standup skills]* *(ClickUp)*
- [ ] **Sentiment Scout** - Tone analysis across interactions, satisfaction trending, pulse surveys, negative trend alerts, competitive sentiment. *[Researcher + sentiment skills]* *(ClickUp)*
- [ ] **Workspace Auditor** - Stale detection, duplicate finder, storage audit, permission review, archive recommendations, scheduled reports. *[Scout + audit skills]* *(ClickUp)*

#### Intelligence Personas
- [ ] **Historian / Archivist** - Manages the long-term evolution of the knowledge graph. Prunes outdated information, archives old project states, creates summaries of historical agent decisions, and can answer questions about "why" a decision was made weeks or months ago by tracing its origins in memory. *[Embedder + Scout + Researcher + memory skills]* *(Suggested Enhancement)*
- [ ] **Wiki Keeper** - Flag outdated docs, auto-update when code changes, link checking, coverage analysis, style enforcement. *[Scout + Researcher + wiki skills]* *(ClickUp)*
- [ ] **Research Analyst** - Deep multi-source research (SearXNG, Scholar, arXiv), source validation, comparison tables, evidence scoring, structured reports, real-time monitoring. *[Researcher + research skills]* *(Original, ClickUp)*
- [ ] **Fact Checker** - Claim extraction, multi-source verification, evidence scoring (strong/weak/conflicting), source credibility assessment, real-time checking in chat. *[Researcher + fact-check skills]* *(ClickUp)*
- [ ] **Competitive Intel** - Competitor profiles, website change monitoring, social listening, product comparison matrices, news alerts, strategic reports. *[Browser + Researcher + intel skills]* *(ClickUp)*

#### Writing Personas
- [ ] **Copywriter** - Brand voice matching, channel-adapted content, A/B variants, hooks and CTAs, SEO keyword integration, content library. *[Researcher + copywriting skills]* *(ClickUp)*
- [ ] **Proofreader** - Grammar/spelling, style guide enforcement, readability scoring, consistency checking, tone matching, track changes. *[Researcher + proofreading skills]* *(ClickUp)*
- [ ] **Release Notes Writer** - Pull completed tickets/PRs from Jira/GitHub/Linear, categorize, translate to user-friendly language, distribute across channels. *[Researcher + Coder + release skills]* *(ClickUp)*
- [ ] **PRD Writer** - Brief-to-PRD expansion with user stories, competitive research, technical feasibility flags, stakeholder input synthesis. *[Researcher + product skills]* *(ClickUp)*
- [ ] **Translator** - 100+ languages via LLM translation, context-aware, per-project glossaries, batch translation, localization beyond words. *[Researcher + translation skills]* *(ClickUp)*

#### Creative & Dev Personas
- [ ] **Image Creator** - ComfyUI, Stable Diffusion, DALL-E, Flux. Style bibles, multi-channel asset adaptation, batch generation, brand asset management. *[Operator + image skills]* *(Original, ClickUp)*
- [ ] **Diagram Maker** - Draw.io (MCP), Mermaid, PlantUML, D2. Flowcharts, ERDs, sequence diagrams, architecture diagrams. Code-to-diagram auto-generation. *[Coder + diagram skills]* *(Original)*
- [ ] **Game Developer** - Unreal, Unity, Godot 4, Roblox, Defold, Bevy. Asset generation, level design, game logic, monetization setup, automated playtesting. *[Coder + game dev skills]* *(Original)*
- [ ] **Streaming Manager** - Twitch, YouTube Live, TikTok Live. OBS scene config, chat moderation, highlight clipping, VOD processing, analytics. *[Browser + Operator + streaming skills]* *(Original)*
- [ ] **Full-Stack Developer** - Scaffold projects, generate components, refactor, test. Go, JS/TS, Python, Rust, Java, Swift, Kotlin. Lint-on-save via Doctor. *[Coder + framework skills]* *(Original)*

#### Security Personas
- [ ] **Pen Tester** - Port scanning (nmap), OWASP Top 10, CVE lookup, XSS/SQLi/CSRF testing, API security, SSL audit, compliance checks (NIST, PCI DSS), scheduled scans. *[Operator + Browser + security skills]* *(Agent Zero, Original)*
- [ ] **ECAM/Security Systems** - Camera systems (Axis, GeoVision, Avigilon, Ubiquiti), NVR management, access control, MSU trailers (Morningstar, Cradlepoint), Zabbix monitoring, re-IP workflows. *[Operator + Browser + ECAM skills]* *(Original)*

#### Media Personas
- [ ] **Transcription Agent** - Universal video transcription: YouTube, TikTok, Instagram, Twitter, Facebook, Twitch. Subtitle extraction via iTaK Media, Whisper fallback, speaker ID, batch processing. *[Operator + Researcher + transcription skills]* *(Original)*

#### Platform Personas
- [ ] **Automation Builder** - Build cross-app workflows on: Zapier, Make, n8n, Node-RED, IFTTT, UiPath, Power Automate, ServiceNow, Workato, Gumloop, Lindy AI. Workflow creation, templates, cross-platform migration. *[Browser + Operator + automation skills]* *(Original)*
- [ ] **Google Workspace** - Gmail, Drive, Docs, Sheets, Calendar, Meet, Contacts, Tasks, Forms, Keep, Chat, YouTube Studio, Search Console, Ads. Uses [gogcli](https://github.com/steipete/gogcli) as the native CLI tool layer (JSON output, command allowlisting for sandboxed runs, OS keyring credential storage, multi-account support). Full CRUD + automation without browser overhead. *[Operator + gogcli + Google skills]* *(Original)*
- [ ] **Microsoft 365** - Outlook, Teams, Office, OneDrive, SharePoint, Azure, Power BI, Dynamics 365, Active Directory. *[Browser + Operator + Microsoft skills]* *(Original)*
- [ ] **Apple Ecosystem** - macOS control, Shortcuts, iCloud, Xcode, TestFlight, App Store Connect. *[Operator + Apple skills]* *(Original)*
- [ ] **Android Controller** - ADB control, app management, notification relay, file transfer, screen mirroring (scrcpy), Tasker integration, iTaK Vision bridge. *[Operator + Android skills]* *(Original)*
- [ ] **Windows Admin** - PowerShell, Registry, Task Scheduler, Services, WSL, Active Directory, Event Log, software management. *[Operator + Windows skills]* *(Original)*
- [ ] **Chromebook Manager** - Chrome OS, Android app sideloading, Crostini Linux container, extensions, enterprise fleet management. *[Operator + ChromeOS skills]* *(Original)*
- [ ] **Linux Admin** - Package management (apt/yum/pacman/snap), systemd, cron, user management, firewall, disk management, performance tuning. Auto-detect distro. *[Operator + Linux skills]* *(Original)*

#### Smart Home Personas
- [ ] **Home Automation** - Home Assistant hub, Philips Hue lighting, thermostat (Nest/Ecobee), 8Sleep, smart locks, sensors, routines ("Good morning", "Goodnight"), energy dashboard. *[Operator + smart home skills]* *(OpenClaw)*
- [ ] **Music Controller** - Spotify, Apple Music, Sonos multi-room, YouTube Music, Shazam recognition, podcast management, ambient modes (focus, sleep). *[Operator + music skills]* *(OpenClaw)*

#### Infrastructure Personas
- [ ] **DevOps Engineer** - Docker, Kubernetes, CI/CD (GitHub Actions, GitLab CI), IaC (Terraform, Pulumi, Ansible), cloud provisioning (AWS/GCP/Azure), SSL/DNS, zero-downtime deploys. *[Operator + Coder + devops skills]* *(Recommendation)*
- [ ] **Data Pipeline Engineer** - ETL/ELT, format conversion (CSV/JSON/Parquet), scheduled pipelines, database connectors (PostgreSQL, BigQuery, DuckDB), data validation, freshness monitoring. Consider [Vaex](https://github.com/vaexio/vaex) for out-of-core processing on million/billion-row datasets via mmap. *[Operator + Coder + data skills]* *(Recommendation)*
- [ ] **Compliance Auditor** - License scanning (SPDX), accessibility (WCAG 2.1), GDPR data mapping, SOC 2 controls, auto-generated compliance reports. *[Scout + Researcher + compliance skills]* *(Recommendation)*
- [ ] **Incident Responder** - Anomaly detection, severity triage, incident channel creation, status page updates, runbook execution, post-mortem generation. *[Operator + Researcher + incident skills]* *(Recommendation)*
- [ ] **Migration Specialist** - Database migrations, framework upgrades, dependency updates, infrastructure moves, rollback plans, pre/post test verification. *[Coder + Operator + migration skills]* *(Recommendation)*
- [ ] **Cost Optimizer** - LLM token tracking, cloud spending analysis, unused resource detection, auto-downgrade models for simple tasks, savings recommendations. *[Researcher + cost skills]* *(Recommendation)*

#### Utility Personas
- [ ] **Onboarding Guide** - First-run setup wizard, capability tour, first project setup, template suggestions, progressive feature disclosure. *[Researcher + onboarding skills]* *(Recommendation)*
- [ ] **Canvas Designer** - Drag-and-drop workflow building, A2UI rendering, collaborative whiteboarding, export to YAML/Mermaid. *[Browser + canvas skills]* *(OpenClaw Canvas)*
- [ ] **Voice Assistant** - Wake word, speech-to-text (Whisper), text-to-speech (iTaK Voice), continuous conversation, voice macros, multi-language. *[Researcher + voice skills]* *(OpenClaw)*
- [ ] **Scheduler** - Cron-style scheduled tasks, natural language ("every Monday at 9am"), dependency chains, calendar awareness, retry policies. *[Operator + scheduler skills]* *(Recommendation)*
- [ ] **Credential Manager** - 1Password, Bitwarden, HashiCorp Vault, built-in encrypted store. Runtime injection, rotation, audit, emergency revoke. *[Operator + credential skills]* *(Recommendation)*
- [ ] **Backup Manager** - Auto-backup knowledge graphs, memories, configs, project history. Scheduled retention, selective restore, machine-to-machine migration. *[Operator + backup skills]* *(Recommendation)*
- [ ] **Git Operator** - Commit, push, pull, rebase, merge, PRs (GitHub/GitLab/Bitbucket), code review, conflict resolution, changelog generation, git hooks. *[Operator + Coder + git skills]* *(Original)*
- [ ] **Database Admin** - PostgreSQL, MySQL, SQLite, MongoDB, Redis, Neo4j, DuckDB. Schema management, migrations, natural language to SQL, EXPLAIN analysis, backup/restore, test data seeding. *[Operator + Coder + database skills]* *(Original)*

#### Industry Personas (Skill Packs via iTaK Hub)
Specialized knowledge and workflows for specific industries. Each builds on top of Marketing + Sales + Customer Service personas.

- [ ] **Pest Control** - Seasonal campaigns (termite/mosquito/rodent), before/after photos, Google Business reviews, local SEO, lead gen landing pages, technician dispatch. *[+ pest control skills]* *(Original)*
- [ ] **Real Estate** - Property listings, MLS sync, virtual tours, open house marketing, market analysis, client pipeline, closing checklists. *[+ real estate skills]* *(Recommendation)*
- [ ] **Restaurant** - Menu management, food photography, Yelp/Google reviews, OpenTable reservations, seasonal marketing, health inspection prep. *[+ restaurant skills]* *(Recommendation)*
- [ ] **Construction** - Project estimation, permit tracking, subcontractor coordination, OSHA safety checklists, progress photos, invoicing. *[+ construction skills]* *(Recommendation)*
- [ ] **Healthcare** - HIPAA-compliant communication, appointment scheduling, patient follow-up, insurance verification, medical record summaries. *[+ healthcare skills]* *(Recommendation)*
- [ ] **Legal** - Contract drafting/review, case timeline management, legal research, client intake, billing/time tracking, regulatory monitoring. *[+ legal skills]* *(Recommendation)*
- [ ] **Education** - Curriculum planning (Common Core aligned), lesson plans, quiz/test creation, auto-grading, student progress, parent communication. *[+ education skills]* *(Recommendation)*
- [ ] **Fitness** - Workout plans, nutrition tracking, client progress photos, class scheduling, wearable sync (Apple Watch, Fitbit, Garmin). *[+ fitness skills]* *(Recommendation)*


## 4. Agent Scaling & Parallelism

- [ ] **Worker Spawning** - Managers spin up N workers for parallel execution. E.g. web agent spins up 20 workers for 20 pages simultaneously. *(Original)*
- [ ] **Parallel Execution** - Workers run concurrently, report to their manager, manager reports to boss. *(Original)*
- [ ] **Dynamic Agent Creation from Chat** - User creates new agents via chat or dashboard. *(Original)*
- [ ] **No Permission Required** - Agents don't ask permission. They get the job done and report what they did. *(Original - notes.md)*
- [ ] **Agent Analytics** - Measure productivity across agents, monitor trends, spot top performers. Usage stats per agent. *(ClickUp Super Agents)*
- [ ] **Multi-User / Team Mode** - Multiple users on one iTaK Agent instance. Role-based access, agent ownership (my agents vs shared), shared knowledge graph with per-user privacy boundaries. *(Recommendation)*

## 5. Memory System

- [x] **Session Memory** - Sliding window of recent messages. *(Agent Zero)*
- [x] **Auto-Reflect** - Agent reflects on completed tasks. *(Agent Zero)*
- [x] **Auto-Entities** - Tracks mentioned entities. *(Agent Zero)*
- [x] **Session Workspace** - Per-session working directory. *(Original)*
- [ ] **Per-Agent Independent Memory** - Each manager has isolated memory. Only results go back to boss. *(Original)*
- [ ] **Knowledge Graph** - Persistent graph memory. Options: [Cayley](https://github.com/cayleygraph/cayley) (Go-native), [Dgraph](https://github.com/dgraph-io/dgraph) (Go-native). Viewable in dashboard. Ships with framework. *(OpenClaw, Original - notes.md)*
- [ ] **Embedded Recall** - Vector similarity search against past conversations/facts. *(OpenClaw)*
- [ ] **Episodic Memory** - Short-term, long-term, and episodic memory layers. Remember what happened, when, and in what context. *(ClickUp Super Agents "Human-level Memory")*
- [ ] **Live Intelligence** - Actively monitors all context to capture and update knowledge bases for people, teams, projects, decisions. Real-time 2-way syncing engine. *(ClickUp Super Agents)*
- [ ] **Infinite Knowledge** - Proprietary real-time syncing with retrieval from fine-tuned embeddings. Enterprise search from connected knowledge across 50+ apps. *(ClickUp Super Agents)*

### BMAD-Inspired Context Management
- [ ] **Context Chain (Document-Driven Handoffs)** - Each workflow phase produces a persistent document that becomes the context for the next agent. Product Brief -> PRD -> Architecture Doc -> Epics/Stories -> Implementation. Managers read upstream docs instead of full chat history. Solves the "agents forget what was decided earlier" problem by encoding decisions in files that feed downstream. The single most impactful orchestration pattern. *(BMAD Method)*
- [ ] **Project Constitution (`project-context.yaml`)** - Auto-generated file on project setup that captures tech stack, language, framework, package manager, coding conventions, naming patterns, error handling style, lint config, and architectural decisions. Scout scans the codebase to detect patterns. Every agent reads this file before doing anything. Doctor knows which linters to run, Coder matches existing style, Builder follows conventions when scaffolding. Replaces ad-hoc "read the codebase first" instructions. *(BMAD Method)*

## 6. LLM Provider System

- [x] **42-Provider Catalog** - All major providers with API endpoints pre-configured. *(Original)*
- [x] **Auto-Discovery** - Calls `/models` on each keyed provider at startup. *(Original)*
- [x] **FailoverClient** - Tries providers in sequence on failure. *(Original)*
- [x] **BudgetClient** - Token spending limits with auto-fallback to cheaper model. *(Original)*
- [/] **iTaK Gateway** - Standalone/embeddable LLM gateway. *(Original, LiteLLM, BricksLLM, Bifrost, Instawork)*
  - [x] Provider adapters (4 API formats: OpenAI, Anthropic, Google, Cohere)
  - [x] Priority/RoundRobin/Latency routing
  - [x] Circuit breaker, rate limiter, content guardrails, cost tracker
  - [x] SSE streaming + standalone CLI binary (9.76 MB)
  - [x] WebTransport support *(quic-go/webtransport-go)*
  - [x] Admin API (provider management, usage reports, circuit breakers)
- [ ] **Optimized Orchestration** - Route to the best model based on intent. Simple queries go to fast/cheap models, complex reasoning goes to powerful models. Auto-classification. *(ClickUp Super Agents "BrainGPT")*
- [ ] **Self-Learning Routing** - Routes improve over time based on success/failure/cost data. Continuous optimization of model selection. *(ClickUp Super Agents)*

## 7. Self-Healing & Monitoring (iTaK Beat)

- [ ] **30-Minute Health Loop** - Doctor checks framework health on timer. *(OpenClaw "doctor")*
- [ ] **Error-Triggered Activation** - Log error auto-triggers doctor. *(Original)*
- [ ] **Boss Halt/Resume** - Boss pauses on doctor activation. Doctor sends thumbs-up when fixed. *(Original)*
- [ ] **Fix Memory** - Doctor logs what fixed each error. Same error = instant recall. *(Original)*
- [ ] **Pre-made Health Scripts** - Framework checks, lint per language, security scans. *(Original)*
- [ ] **Self-Heal Prompts** - Main agent can detect errors and route them to doctor automatically. *(Original - notes.md)*
- [ ] **Nudge Feature** - Poke agent if stuck. *(Agent Zero)*
- [ ] **Linter Integration** - Auto-detect project language and run appropriate linter (golangci-lint, eslint, stylelint, pylint, etc.) after every code change. Results feed into Doctor. *(Original - tasks.md, VS Code stylelint)*
- [ ] **Semantic Code Review** - Goes beyond lint. Doctor compares implementation against the story file's acceptance criteria and the `project-context.yaml` conventions. Checks that architectural patterns are followed, not just syntax. Distinguishes between "code quality issues" (Doctor auto-fixes) and "design drift issues" (flags for Boss to decide). Verifies RWD/PWA/a11y compliance against the mandatory standards. Feeds into the Consensus Verification Loop when enabled. *(BMAD Method)*

## 8. Offline Mode & Local Model Marketplace

- [ ] **Offline Toggle** - Settings toggle for fully offline operation. *(Original)*
- [ ] **Ollama Integration** - Auto-detect local Ollama instance, auto-install if missing. Pull models via Ollama CLI. *(Agent Zero)*
- [ ] **Bundled Embed Model** - Ships with `qwen3-embedding` by default *(pending test vs `nomic-embed-text-v2-moe`)*. *(Original)*
- [ ] **Install-Time Model Picker** - During first run, user selects their hardware tier and picks models from the marketplace. iTaK Agent pulls them via Ollama automatically. *(Original)*
- [ ] **Model Marketplace UI** - Dashboard page showing all available local models organized by role. One-click install/remove. Size, RAM requirements, and quality ratings displayed. *(Original)*
- [ ] **Hardware Auto-Detection** - Detect CPU cores, RAM, GPU presence at startup. Auto-recommend appropriate model tier. *(Original)*
- [ ] **Go Offline Flow** - User downloads desired models while online, then toggles offline mode. iTaK Agent switches all routing to local models. *(Original)*

### Local Model Catalog (Curated)

#### Embedding Models (pick one, always loaded)
| Model | Size | Quality | Best For |
|-------|------|---------|----------|
| `qwen3-embedding` | ~600MB | Excellent | **Default choice.** Latest Qwen3 embeddings. |
| `nomic-embed-text-v2-moe` | ~275MB | Excellent | MoE architecture, very efficient on CPU. Smallest footprint. |
| `bge-large` | ~670MB | Great | Battle-tested BAAI embedding. Reliable fallback. |
| `mxbai-embed-large` | ~670MB | Great | mixedbread embedding. Strong retrieval performance. |
| `embeddinggemma` | ~800MB | Good | Google quality but larger than alternatives. |

#### Chat / Main Brain Models (pick one based on hardware)
| Model | Params | Size | Min RAM | Tier | Notes |
|-------|--------|------|---------|------|-------|
| `qwen3` 0.6B | 0.6B | ~400MB | 4GB | Nano | Fast router/classifier only. Too small for real tasks. |
| `LFM2` 1.2B | 1.2B | ~800MB | 4GB | Nano | **State-space model.** Fastest CPU inference. Made for edge. |
| `granite-4.0-nano` | 1-2B | ~1.2GB | 4GB | Nano | IBM's latest. Excellent tool-calling at tiny size. |
| `qwen3` 1.7B | 1.7B | ~1.2GB | 4GB | Lite | Solid balance of smart + small. Good tool calling. |
| `ministral-3` | 3B | ~2GB | 8GB | Lite | Mistral's edge model. Good instruction following. |
| `stablelm-zephyr` | 3B | ~2GB | 8GB | Lite | Decent chat. Older but proven. |
| `qwen3` 4B | 4B | ~2.5GB | 8GB | Standard | **Best brain under 3GB.** Smartest small model. |
| `qwen3` 8B | 8B | ~4.5GB | 16GB | Standard | Strong all-rounder. Needs 16GB RAM. |
| `deepseek-r1` 8B distill | 8B | ~4.5GB | 16GB | Standard | Reasoning-focused. Good for complex planning. |
| `qwen3.5` 7B | 7B | ~4GB | 16GB | Pro | Latest Qwen generation. Best quality/size if RAM allows. |
| `qwen3` 14B | 14B | ~8GB | 32GB | Pro | Excellent quality. Needs serious RAM. |

#### Coding Models (swap in when coding agent is active)
| Model | Params | Size | Notes |
|-------|--------|------|-------|
| `granite-4.0-nano` | 1-2B | ~1.2GB | IBM tool-calling + code. Best tiny coder. |
| `qwen2.5-coder` 3B | 3B | ~2GB | Purpose-built for code gen. |
| `qwen2.5-coder` 7B | 7B | ~4.5GB | Strong code gen. Needs 16GB. |
| `deepseek-r1` 8B distill | 8B | ~4.5GB | Reasons through code problems step by step. |

#### Vision Models (for iTaK Vision / image understanding)
| Model | Params | Size | Notes |
|-------|--------|------|-------|
| `LFM2-VL` | 1-2B | ~1.5GB | Liquid vision. Most CPU-efficient. |
| `qwen3-vl` (smallest) | 2-4B | ~2-3GB | Latest Qwen vision. Good accuracy. |

#### Reasoning / Thinking Models (for complex planning)
| Model | Params | Size | Notes |
|-------|--------|------|-------|
| `deepseek-r1` 1.5B distill | 1.5B | ~1GB | Chain-of-thought but limited by size. |
| `deepseek-r1` 8B distill | 8B | ~4.5GB | Strong reasoning. 16GB required. |
| `qwen3` 4B (thinking mode) | 4B | ~2.5GB | Qwen3's built-in thinking mode. |

### Hardware Tier Presets
| Tier | RAM | Example Hardware | Recommended Stack |
|------|-----|-----------------|-------------------|
| **Nano** (4GB) | 4-8GB | Raspberry Pi 5, old laptops | LFM2 1.2B + nomic-embed-v2-moe |
| **Lite** (8GB) | 8-12GB | Budget desktops, Chromebooks | Qwen3 4B + qwen3-embedding |
| **Standard** (16GB) | 12-16GB | **Dell OptiPlex 7060 (Skynet)** | Qwen3 8B + Granite Code + qwen3-embedding |
| **Pro** (32GB+) | 32GB+ | Workstations, gaming PCs | Qwen3 14B + Qwen2.5-Coder 7B + qwen3-embedding |

## iTaK Browser

### 9. iTaK Browser (AI-Native Browser Engine)

Go-native browser automation CLI for agents. Inspired by [Vercel agent-browser](https://github.com/vercel-labs/agent-browser) architecture, rebuilt entirely in Go. Uses Snapshot+Refs pattern for 93% token reduction vs raw DOM dumps.

**Research:** [rtrvr.ai](https://www.rtrvr.ai/) - AI-native browser retrieval platform. Study their approach to structured web data extraction for agent consumption.

### Architecture
- [ ] **Go CLI** - Fast native binary. Parses commands, communicates with daemon. Replaces agent-browser's Rust CLI. *(agent-browser pattern)*
- [ ] **Go Daemon** - Manages browser instance via CDP (Chrome DevTools Protocol). Persists between commands for fast sequential operations. *(agent-browser pattern)*
- [ ] **CDP-Native (Default)** - Direct Chrome DevTools Protocol via `chromedp` (Go-native). No Node.js, no Playwright required. *(agent-browser native mode, chromedp)*
- [ ] **Playwright Fallback** - For complex pages or Firefox/WebKit support, fall back to Playwright via shell. *(agent-browser pattern)*

### Snapshot + Refs (Core Innovation)
The key to making small LLMs work with browsers. Instead of dumping 50KB of HTML, generate a compact accessibility tree with refs:

```
# Agent asks for snapshot, gets:
- heading "Example Domain" [ref=e1] [level=1]
- button "Submit" [ref=e2]
- textbox "Email" [ref=e3]
- link "Learn more" [ref=e4]

# Agent says: click @e2, fill @e3 "test@example.com"
# ~200 tokens instead of 10,000
```

- [ ] **Accessibility Snapshots** - Generate compact text-based tree of all interactive elements with unique refs. 93% fewer tokens than raw DOM. *(agent-browser)*
- [ ] **Ref-Based Interaction** - Click, fill, hover, get text by ref ID. Deterministic, fast, no DOM re-query. *(agent-browser)*
- [ ] **Annotated Screenshots** - Screenshots with numbered element labels overlaid. Vision models can reference elements by number. *(agent-browser)*
- [ ] **JSON Agent Mode** - All commands output structured JSON for machine parsing. `--json` flag on every command. *(agent-browser)*

### Core Commands (50+)
- [ ] **Navigation** - open, close, back, forward, reload. *(agent-browser)*
- [ ] **Interaction** - click, dblclick, fill, type, press, hover, select, check, uncheck, scroll, drag, upload. *(agent-browser)*
- [ ] **Data Extraction** - get text, get html, get value, get attr, get title, get url, get count, get box, get styles. *(agent-browser)*
- [ ] **State Checks** - is visible, is enabled, is checked. *(agent-browser)*
- [ ] **Waiting** - Wait for selector, time, text, URL pattern, network idle, JS condition. *(agent-browser)*
- [ ] **Mouse Control** - move, down, up, wheel for pixel-precise control. *(agent-browser)*
- [ ] **Keyboard** - Full keyboard emulation. Key combos, inserttext, keydown/keyup. *(agent-browser)*
- [ ] **Capture** - screenshot, screenshot --annotate, snapshot, pdf, eval (JavaScript). *(agent-browser)*

### Semantic Locators
- [ ] **Find by Role** - `gobrowser find role button click --name "Submit"`. ARIA-aware element discovery. *(agent-browser)*
- [ ] **Find by Text/Label** - `gobrowser find text "Sign In" click`. Natural language element finding. *(agent-browser)*
- [ ] **Find by Placeholder/Alt/TestID** - Multiple strategies for locating elements. *(agent-browser)*

### Sessions & Persistence
- [ ] **Persistent Profiles** - Dedicated browser profiles per agent/task. Cookies, storage, history persist across sessions. *(agent-browser)*
- [ ] **Session Persistence** - Save and restore browser state (cookies, storage, open tabs). *(agent-browser)*
- [ ] **State Encryption** - Encrypt persisted session data at rest. API keys and auth tokens stay safe. *(agent-browser)*

### Stealth & Anti-Detection
- [ ] **Stealth Mode** - Evade bot detection (headless fingerprinting, navigator.webdriver, etc.). *(Browserbase patterns)*
- [ ] **Proxy Support** - HTTP/SOCKS5 proxy per session. Rotate IPs for scraping. *(Original)*
- [ ] **Custom User-Agent** - Rotate user agents. Mobile/desktop emulation. *(Original)*

### Advanced
- [ ] **Browser Streaming** - WebSocket-based live preview of browser viewport. Watch agents browse in real-time from iTaK Dashboard. *(agent-browser streaming)*
- [ ] **Page Diff** - Compare DOM snapshots between actions. Track what changed. Useful for testing. *(agent-browser diff)*
- [ ] **Tab/Window Management** - Open, close, switch tabs. Multi-tab workflows. *(agent-browser)*
- [ ] **Network Interception** - Monitor, mock, block network requests. *(Playwright pattern)*
- [ ] **CDP Connect** - Attach to an already-running browser via CDP port. Reuse existing sessions. *(agent-browser connect)*
- [ ] **Teaching Mode** - Record agent actions as interactive tutorials for the Teacher Agent. *(Original)*
- [ ] **DOM Extraction** - Navigate and extract structured data via DOM. *(Original)*

## iTaK Vision

### 10. iTaK Vision (Screen Automation Agent)

Inspired by [Open-AutoGLM](https://github.com/THUDM/Open-AutoGLM). Rebuilt from scratch in Go.

- [ ] **Screen Capture Engine** - Take screenshots of desktop/mobile screens. *(Open-AutoGLM)*
- [ ] **Vision Model Integration** - Send screenshots to vision LLM (Gemini, GPT-4V, Qwen-VL) to understand UI state. *(Open-AutoGLM)*
- [ ] **Action Planner** - Plan next click/type/scroll action based on screen understanding + user goal. *(Open-AutoGLM)*
- [ ] **Desktop Executor** - Execute planned actions on Windows/Linux/Mac via OS-native APIs. *(Original)*
- [ ] **ADB Bridge** - Execute actions on Android devices via ADB (Android Debug Bridge). *(Open-AutoGLM)*
- [ ] **Teaching Mode Integration** - Record action sequences as interactive tutorials for the Teacher Agent. *(Original)*

## iTaK Teach

### 11. iTaK Teach (Interactive Teaching & Training Platform)

AI-powered teaching system that uses DOM manipulation, screen recording, and interactive overlays to teach users anything - from coding to Figma to database management. The Teacher Agent doesn't just tell you how to do something, it shows you on your actual screen.

### DOM-Based Interactive Walkthroughs
- [ ] **Element Highlighting** - Inject CSS overlays to highlight specific elements on any web page. Pulsing borders, spotlights, dimming the rest of the page. "Click this button right here." *(Original)*
- [ ] **Tooltip Overlays** - Inject floating tooltips/popovers next to UI elements with step-by-step instructions. Arrow pointing to the element, text explaining what to do. Like Intro.js but agent-generated. *(Original)*
- [ ] **Click-Through Tutorials** - Step-by-step guided walkthroughs. User clicks "Next" to advance, or agent waits for user to perform the action. Validates the user did it correctly. *(Original)*
- [ ] **Form Fill Demos** - Agent fills out forms in slow motion, showing the user what goes where. Then clears and lets the user try. *(Original)*
- [ ] **Before/After Snapshots** - Capture DOM state before and after an action. Show the user exactly what changed and why. *(Original)*

### Screen Recording & Narration
- [ ] **Action Recording** - Record agent's browser/desktop actions as video. Every click, scroll, and keypress captured. *(Original)*
- [ ] **AI Voice Narration** - Generate voice-over narration explaining each step as the agent performs it. Uses iTaK Torch local TTS or cloud TTS. *(Original)*
- [ ] **Annotated Recordings** - Overlay mouse cursor highlights, click indicators, and text callouts on recordings. *(Original)*
- [ ] **GIF Generation** - Auto-generate short GIFs of specific procedures. Perfect for documentation and quick reference. *(Original)*
- [ ] **Tutorial Export** - Export tutorials as MP4, GIF, interactive HTML, or Markdown with embedded screenshots. Shareable via link. *(Original)*

### Multi-Platform Teaching
The Teacher Agent doesn't just work in browsers. It teaches across platforms:

| Platform | How It Teaches |
|----------|---------------|
| **Browser** | DOM overlays, element highlighting, form fill demos |
| **Desktop Apps** | iTaK Vision screen capture + annotated screenshots. "Click here on Photoshop's toolbar" |
| **Mobile** | ADB bridge + screen mirroring. Guides user through phone apps |
| **CLI/Terminal** | Shows commands with syntax highlighting, explains flags, runs examples in sandbox |
| **Databases** | Visual query building, schema diagrams, step-by-step migration walkthroughs |
| **APIs** | Interactive request builder, shows headers/body/response with explanations |
| **Code** | Pair programming with line-by-line annotations, live code execution |

- [ ] **Platform Detection** - Auto-detect what the user is trying to learn (browser, desktop app, CLI, code) and switch teaching mode. *(Original)*
- [ ] **Cross-Platform Tutorials** - Tutorials that span multiple platforms. "First, create the database (CLI), then build the API (code), then test it (browser)." *(Original)*

### Adaptive Learning
- [ ] **Mistake Detection** - When user does something wrong during a tutorial, agent notices and course-corrects. "Almost! You clicked the wrong tab. Try the one to the left." *(Original)*
- [ ] **Difficulty Adjustment** - If user breezes through steps, speed up. If they struggle, slow down and add more detail. *(Original)*
- [ ] **Learning Style Detection** - Track whether user learns better from seeing (video), reading (text), or doing (interactive). Adapt tutorial format. *(Original)*
- [ ] **Progress Tracking** - Track which tutorials completed, skills learned, areas of struggle. Build a learning profile per user. *(Original)*

### Curriculum Generation
- [ ] **Learning Paths** - Generate full curricula: "Learn React in 2 weeks" with ordered lessons, exercises, and projects. *(Original)*
- [ ] **Prerequisite Detection** - If user tries an advanced topic, auto-detect prerequisite gaps. "Before learning GraphQL, you should know REST APIs. Want me to teach that first?" *(Original)*
- [ ] **Project-Based Learning** - Generate real projects that teach concepts incrementally. Each step builds on the last. *(Original)*
- [ ] **Quiz/Assessment** - After each lesson, generate quiz questions to test understanding. Adaptive difficulty. *(Original)*
- [ ] **Certification Badges** - Issue skill badges when user completes a learning path. Viewable in iTaK Dashboard. *(Original)*

### Pair Programming Mode
- [ ] **Live Code Watch** - Agent watches user code in real-time. Offers suggestions, catches errors before they happen. *(Original)*
- [ ] **Rubber Duck Mode** - User explains their code to the agent. Agent asks clarifying questions that help the user find their own bugs. *(Original)*
- [ ] **Code Review Tutor** - Agent reviews user's code and explains WHY something is wrong, not just WHAT. Teaches principles, not just fixes. *(Original)*
- [ ] **Refactoring Demonstrations** - Agent takes user's working code and shows them how to improve it step-by-step. *(Original)*

### Tutorial Marketplace (iTaK Hub)
- [ ] **Community Tutorials** - Users create and share tutorials on iTaK Hub. Rate, review, and fork tutorials. *(Original)*
- [ ] **Agent-Generated Docs** - Agent auto-generates documentation for any project it works on. Tutorials for the code it writes. *(Original)*
- [ ] **Onboarding Packs** - Pre-built tutorial packs for common setups: "New developer onboarding", "DevOps basics", "iTaK Agent power user". *(Original)*

## iTaK Voice

### Voice & TTS Engine (iTaK Voice)
The voice layer that powers narrated tutorials, voice agents, and video content:

| Engine | Type | Quality | Speed | Notes |
|--------|------|---------|-------|-------|
| **Piper TTS** | Local | Good | Very Fast | Go-compatible via CLI. Ships with iTaK Agent. No internet needed. |
| **Kokoro TTS** | Local | Great | Fast | High-quality neural TTS. Runs on CPU. |
| **Edge TTS** | Cloud (Free) | Great | Fast | Microsoft's free TTS API. No API key needed. |
| **ElevenLabs** | Cloud (Paid) | Excellent | Fast | Best voice quality. Voice cloning. |
| **OpenAI TTS** | Cloud (Paid) | Excellent | Fast | Multiple voices, natural speech. |
| **Google Cloud TTS** | Cloud (Paid) | Great | Fast | WaveNet voices. |

- [ ] **Local TTS (Default)** - Ship with Piper TTS for zero-dependency voice generation. Works offline. Multiple voices and languages. *(Original)*
- [ ] **Cloud TTS Fallback** - For higher quality, route to ElevenLabs, OpenAI TTS, or Edge TTS (free). Auto-select based on quality needs. *(Original)*
- [ ] **Voice Cloning** - Clone a specific voice from a sample. User records 1 minute of audio, iTaK Voice learns the voice for all future content. Via ElevenLabs or local Coqui. *(Original)*
- [ ] **Voice Profiles** - Save named voice profiles: "Professional Male", "Friendly Female", "Brand Voice". Switch per project. *(Original)*
- [ ] **Multi-Language TTS** - Generate narration in 30+ languages. Auto-detect from script language or user preference. *(Original)*
- [ ] **SSML Support** - Control pacing, emphasis, pauses, and pronunciation via Speech Synthesis Markup Language. *(Standard)*
- [ ] **Real-time Streaming TTS** - Stream audio as it generates. Don't wait for full generation. For live voice agent interactions. *(Original)*

### Content Production Pipeline (Topic-to-Video Factory)
Give iTaK Teach a topic. It researches, builds the tutorial, records it with DOM manipulation, narrates with AI voice, edits the video, and publishes to social media. Full autopilot content creation.

**The Pipeline:**
```
User says: "Create a YouTube tutorial on how to deploy a Next.js app to Vercel"
  │
  ├── Step 1: RESEARCH
  │     └── Research Agent gathers current best practices, official docs, common pitfalls
  │
  ├── Step 2: SCRIPT
  │     └── Write a narration script with sections, timestamps, and key points
  │
  ├── Step 3: ENVIRONMENT
  │     └── Open iTaK Browser, set up a clean project, install dependencies
  │
  ├── Step 4: RECORD
  │     └── Execute each step in the browser while screen recording
  │     └── DOM highlighting on key elements (buttons, menus, config fields)
  │     └── Pause between steps for narration sync
  │
  ├── Step 5: NARRATE
  │     └── iTaK Voice generates narration from script, synced to video timestamps
  │
  ├── Step 6: EDIT
  │     └── Add intro/outro, chapter markers, captions, transitions
  │     └── Generate thumbnail via Image Agent
  │     └── Add background music (royalty-free library)
  │
  ├── Step 7: REVIEW
  │     └── Play back for user approval (or auto-approve at Autonomy Level 4)
  │
  └── Step 8: PUBLISH
        └── Upload to YouTube, TikTok, Instagram, Twitter, LinkedIn
        └── Auto-generate title, description, tags, chapters, hashtags
        └── Schedule or publish immediately
```

- [ ] **Topic-to-Script** - Given a topic, Research Agent gathers info, then generates a structured tutorial script with intro, sections, code examples, and outro. *(Original)*
- [ ] **Auto-Environment Setup** - Spin up a clean project environment for the tutorial. Install deps, create files, configure settings - all while recording. *(Original)*
- [ ] **Synchronized Recording** - Record iTaK Browser/iTaK Vision screen while executing tutorial steps. DOM highlights sync with narration timestamps. *(Original)*
- [ ] **AI Narration Sync** - iTaK Voice generates narration timed to screen actions. Natural pacing with pauses for visual comprehension. *(Original)*
- [ ] **Video Editing** - Auto-add intro/outro sequences, chapter markers, smooth transitions between sections, lower-third text callouts. *(Original)*
- [ ] **Auto-Captions** - Generate accurate subtitles/closed captions from the narration. Burn in or attach as .srt file. Multi-language. *(Original)*
- [ ] **Thumbnail Generation** - Image Agent creates eye-catching thumbnails with title text, screenshots, and branding. Multiple variations to A/B test. *(Original)*
- [ ] **Background Music (Copyright-Free)** - Add ambient background music at low volume. Auto-ducks to ~10-15% volume when narration is playing, rises slightly during silent pauses. Only uses royalty-free/Creative Commons sources:
  - **Built-in Library** - Ships with 50+ royalty-free tracks (lo-fi, ambient, upbeat tech) organized by mood
  - **Pixabay Music** - Free API, no attribution required, thousands of tracks
  - **Free Music Archive** - Creative Commons licensed tracks
  - **Incompetech (Kevin MacLeod)** - Classic YouTube tutorial background music, CC-BY
  - **AI-Generated Music** - Generate custom background tracks via Suno, Udio, or local MusicGen model. Unique per video, zero copyright risk
  - **Mood Matching** - Auto-select music mood based on tutorial tone: chill for coding, upbeat for intros, calm for explanations
  - **Seamless Looping** - Tracks loop seamlessly across the full video length with crossfade. No abrupt cuts *(Original)*
- [ ] **SEO Optimization** - Auto-generate YouTube-optimized titles, descriptions, tags, and hashtags. Research trending keywords for the topic. *(Original)*
- [ ] **Multi-Platform Publishing** - Upload final video to YouTube, TikTok (vertical crop), Instagram Reels, Twitter/X, LinkedIn. Platform-specific formatting. *(Original)*
- [ ] **Series Generation** - Create multi-part tutorial series. "Part 1: Setup", "Part 2: Authentication", "Part 3: Deployment". Auto-link and playlist. *(Original)*
- [ ] **Content Calendar** - Schedule content production. "Create 3 tutorials per week on React topics." Agent handles the entire pipeline on schedule. *(Original)*
- [ ] **Analytics Feedback Loop** - Track video performance (views, engagement, retention). Agent learns which topics/styles perform best and adjusts future content. *(Original)*

### Human Realism Mode (No AI Look)
Videos should look like a human recorded them, not a robot. Two modes:
- **Interactive Mode** = DOM highlights, tooltips, overlays (for live tutorials where user is learning)
- **Production Mode** = Clean, natural recording (for YouTube/social media content)

- [ ] **Natural Mouse Movement** - Bezier curve mouse paths, slight overshoot, random micro-pauses. Not pixel-perfect straight lines. *(Original)*
- [ ] **Realistic Typing Speed** - Variable WPM (50-80), occasional brief pauses between words, natural rhythm. Not uniform machine-gun typing. *(Original)*
- [ ] **No Overlay in Production** - DOM highlights, tooltips, and glowing borders are OFF in production mode. Clean screen recording only. *(Original)*
- [ ] **Human Scroll Patterns** - Scroll at natural speed. Sometimes overshoot and scroll back slightly. Read-pause behavior. *(Original)*
- [ ] **Natural Pacing** - Pause before clicking (like a human reading the button). Vary timing between actions. Not perfectly metronomic. *(Original)*
- [ ] **Cursor Warmth** - Slight cursor wobble. Humans don't hold the mouse perfectly still. *(Original)*

### AI Avatar & Digital Twin (iTaK Avatar)
Appear in your own videos without recording yourself. AI-generated talking head using your face and voice.

| Technology | What It Does | Open Source Options |
|-----------|-------------|-------------------|
| **Face Generation** | Create realistic avatar from a single photo | SadTalker, LivePortrait, MuseTalk |
| **Lip Sync** | Sync avatar's mouth to narration audio | Wav2Lip, SyncTalk, MuseTalk |
| **Voice Clone** | Clone your voice from 1-min audio sample | Coqui TTS, OpenVoice, Fish Speech |
| **Full Body** | Generate full presenter with gestures | HeyGen patterns, D-ID patterns |

- [ ] **Avatar from Photo** - Upload one photo of yourself. iTaK Avatar generates a realistic talking head video. Lip-synced to narration. *(Original, SadTalker/LivePortrait)*
- [ ] **Voice Clone Integration** - Record 60 seconds of your voice. iTaK Voice clones it. All future tutorials sound like you. *(Original, OpenVoice/Coqui)*
- [ ] **Picture-in-Picture** - Avatar appears in corner of screen recording, like a webcam. Standard YouTube tutorial layout. *(Original)*
- [ ] **Full-Screen Presenter** - Avatar appears full-screen for intro/outro segments. Switches to screen recording for the tutorial steps. *(Original)*
- [ ] **Gesture Generation** - Avatar uses natural hand gestures while speaking. Not a frozen head. *(Original)*
- [ ] **Multiple Avatars** - Create different presenter personas for different channels/brands. *(Original)*
- [ ] **Green Screen Mode** - Generate avatar with transparent background. Composite onto any scene. *(Original)*
- [ ] **Local Processing (Default)** - Avatar generation runs locally via iTaK Torch + open-source models (SadTalker, Wav2Lip). No cloud dependency. *(Original)*
- [ ] **Cloud Fallback** - For higher quality, route to HeyGen or D-ID APIs. *(Original)*

## iTaK Forge

### 12. iTaK Forge (Live Preview + Container Runtime)

Go-native live preview server and lightweight container runtime for agent workloads. Every project built by iTaK Agent gets a live URL instantly. Separate repo (`iTaK Forge`) but embeddable into iTaK Agent as a library.

### GitHub Integration Pipeline
- [ ] **Auto-Repo Creation** - Git Agent creates a private GitHub repo via API when a new project starts. *(Original)*
- [ ] **Live Push** - Code is committed and pushed to GitHub after each agent iteration. *(Original)*
- [ ] **Webhook Receiver** - GitHub push webhooks trigger automatic builds and deploys in iTaK Forge. *(Original)*
- [ ] **Branch Previews** - Each branch gets its own preview URL. PRs show deploy previews. *(Vercel/Netlify pattern)*

### Live Preview Server
- [ ] **Project Auto-Detection** - Reads `package.json`, `go.mod`, `index.html`, `requirements.txt`, `Dockerfile` to identify project type. *(Original)*
- [ ] **Builder Engine** - Runs appropriate build command per project type (npm run build, go build, pip install, static serve). *(Original)*
- [ ] **Hot Reload** - New pushes trigger automatic rebuild and reload. User sees changes in seconds. *(Original)*
- [ ] **Reverse Proxy** - Routes `project-name.localhost:PORT` to the correct running process. Clean URLs per project. *(Original)*
- [ ] **Preview Dashboard** - Web UI showing all deployed projects, build status, logs, and URLs. *(Original)*
- [ ] **Deploy API** - Simple REST API for iTaK Agent to trigger deploys and check status. One call = one deploy. *(Original)*

### Tier 1: Process Isolation (MVP)
- [ ] **Process Groups** - Each project runs in its own process group. Isolated stdout/stderr. *(Original)*
- [ ] **Temp Workspaces** - Clean temporary directory per build. No cross-project contamination. *(Original)*
- [ ] **Process Manager** - Keeps apps alive, restarts on crash, graceful shutdown. *(Original)*
- [ ] **Port Manager** - Auto-assigns ports from a pool. No conflicts. *(Original)*

### Tier 2: Real Containers (Full Isolation)
- [ ] **Linux Namespaces** - PID, network, mount, UTS namespaces via Go `syscall` package for true container isolation. *(Docker pattern)*
- [ ] **Cgroups** - CPU and memory limits per container. Prevents runaway agents from crashing the host. *(Docker pattern)*
- [ ] **Rootfs Management** - Minimal root filesystems per project type (node-slim, go-slim, python-slim). *(Original)*
- [ ] **Agent Sandboxing** - Every agent worker runs inside a container. Bad code can only destroy its own sandbox. Hardware-enforced safety. *(Original)*
- [ ] **Extension Isolation** - Third-party iTaK Hub extensions run in their own containers. No filesystem/API key access to host. *(Original)*
- [ ] **Snapshot & Restore** - Checkpoint a container's state at any point. Restore for debugging. Pairs with StepLogger for time-travel. *(Original)*
- [ ] **OCI Compatibility** - Can pull base images from Docker Hub/GHCR if needed. *(Standard)*
- [ ] **Windows Support** - Hyper-V isolation or WSL2 bridge for Windows hosts. *(Original)*

### Tier 3: Full Platform
- [ ] **Container Networking** - Private networks between containers for multi-service apps (frontend + backend + database). *(Docker Compose pattern)*
- [ ] **Built-in Image Registry** - Store and cache base images locally. No Docker Hub dependency for common images. *(Original)*
- [ ] **Container-to-Container Communication** - Service discovery so containers can talk to each other by name. *(Docker pattern)*
- [ ] **Parallel Worker Pools** - Spin up N containers for N tasks. 20 browser workers = 20 isolated Chromium instances. *(Original)*
- [ ] **Multi-Language Environments** - Go 1.22, Node 22, Python 3.12 all running simultaneously with zero conflicts. *(Original)*
- [ ] **Disposable Workers** - Fire-and-forget containers that auto-cleanup after task completion. *(Original)*
- [ ] **Resource Dashboard** - CPU/memory/network usage per container. Integrated into iTaK Dashboard. *(Original)*

### Supported Project Types
- [ ] **Static HTML/CSS/JS** - Direct file serving with live reload. *(Original)*
- [ ] **Vite/React/Vue/Next.js** - `npm install` + `npm run build` + serve dist/. *(Original)*
- [ ] **Go Applications** - `go build` + run binary. *(Original)*
- [ ] **Python/Flask/FastAPI** - `pip install` + run server. *(Original)*
- [ ] **Dockerfile Projects** - If Dockerfile exists, build and run as container. *(Original)*

## iTaK Dashboard

### 13. Dashboard (iTaK Dashboard)

- [x] **Real-time Agent Monitoring** - Live feed of agent activity. *(Agent Zero dashboard)*
- [x] **Chat Interface** - Talk to orchestrator from browser. *(Agent Zero, OpenClaw)*
- [x] **Dark Mode** - Professional dark UI. *(OpenFang dashboard)*
- [ ] **Direct Agent Chat** - Tap into any specific manager agent from dashboard. *(Original - notes.md)*
- [ ] **Agent Creation UI** - Create/edit agents from dashboard. *(Original)*
- [ ] **Project Management UI** - Agent Zero-style project setup via dashboard and CLI. *(Agent Zero, Original - notes.md)*
- [ ] **Provider Management** - Add/remove API keys, see health status. *(Original)*
- [ ] **Cost Dashboard** - Token usage graphs, per-provider spending. *(BricksLLM)*
- [ ] **Knowledge Graph Viewer** - Visual graph exploration in dashboard. *(Original - notes.md)*
- [ ] **Log Viewer** - Real-time structured logs with filtering. *(OpenFang)*
- [ ] **iTaK Beat Status** - Health check results, doctor activity. *(Original)*
- [ ] **Private Info Manager** - UI for .env variables: API keys, tokens, provider configs. *(Original - notes.md)*
- [ ] **Kanban Board View** - Drag-and-drop task board for agent work items. Columns: Pending, Running, Done, Failed. Per-agent and per-project views. *(ClickUp Super Agents)*
- [ ] **Agent Analytics Dashboard** - Productivity metrics, usage percentile, top performer ranking, milestone tracking. *(ClickUp Super Agents)*
- [ ] **iTaK Hub Browser** - Browse, search, install, and manage extensions from the dashboard. Rating system and reviews. *(VS Code Marketplace, OpenClaw ClawHub)*
- [ ] **Canvas/Whiteboard** - Visual workspace for building agent workflows, diagramming architectures, and collaborative planning. *(ClickUp Whiteboards, OpenClaw Canvas)*
- [ ] **Draw.io Integration** - Embed draw.io diagrams in dashboard for architecture docs, flowcharts, and ERDs. Via draw.io MCP server. *(Original - tasks.md)*
- [ ] **iTaK Forge Deploy Panel** - View all live preview deployments, build logs, and project URLs directly from dashboard. *(Original)*

## 14. Transport & Connectivity

- [x] **SSE Streaming** - Server-Sent Events for universal compatibility. *(Standard)*
- [x] **WebTransport (HTTP/3)** - Primary transport for Go-to-Go communication. Uses `quic-go/webtransport-go`. Reliable streams + unreliable datagrams. *(Original - notes.md)*
- [ ] **Webhook Support** - For n8n, automation platforms. Inbound/outbound webhooks. *(Original - notes.md)*

## 15. Sequential Thinking Engine

- [ ] **Built-in Sequential Thinking** - Go-native implementation of the Sequential Thinking MCP pattern. Improves reasoning and planning for small models. Chain-of-thought with revision loops. *(Anthropic MCP, Original - notes.md)*
- [ ] **Research**: How does the Sequential Thinking MCP work? Build our own Go version. *(Original - notes.md)*

## 16. MCP System

- [x] **MCP Client** - Connect to external MCP servers. `pkg/mcp/client.go` *(Standard)*
- [ ] **MCP Server** - Expose iTaK Agent tools as MCP server. Written in Go. *(Standard)*
- [ ] **MCP Discovery** - Auto-discover and register from config. *(Standard)*
- [ ] **Bundled MCP Servers** (user activates what they want):
  - [ ] GitHub - repo tasks, code reviews
  - [ ] BrightData - web scraping + data feeds
  - [ ] GibsonAI - serverless SQL management
  - [ ] Notion - workspace + DB automation
  - [ ] Docker Hub - container + DevOps
  - [ ] Browserbase - browser control
  - [ ] Context7 - live code examples + docs
  - [ ] Figma - design-to-code
  - [ ] Reddit - fetch/analyze data
  - [ ] Sequential Thinking - reasoning loops
  - [ ] Draw.io - architecture diagrams, flowcharts, ERDs *(https://github.com/lgazo/drawio-mcp-server)*
  - [ ] 1Password - secure credential access

## 17. Communication Plugins

Expanded to match OpenClaw's 15+ chat provider coverage:

- [ ] **Discord** - Bot integration, chat, commands. *(OpenClaw)*
- [ ] **Telegram** - Bot API via grammY. *(OpenClaw)*
- [ ] **Email** - Send/receive/parse emails. Gmail Pub/Sub triggers. *(OpenClaw)*
- [ ] **WhatsApp** - QR pairing via Baileys. Go CLI via `steipete/wacli`. *(OpenClaw, Original - tasks.md)*
- [ ] **Slack** - Workspace apps via Bolt. *(Standard, OpenClaw)*
- [ ] **Signal** - Privacy-focused via signal-cli. *(OpenClaw)*
- [ ] **iMessage** - Via AppleScript bridge (imsg) or BlueBubbles server. *(OpenClaw)*
- [ ] **Microsoft Teams** - Enterprise support. *(OpenClaw)*
- [ ] **Matrix** - Matrix protocol for decentralized chat. *(OpenClaw)*
- [ ] **Nostr** - Decentralized DMs via NIP-04. *(OpenClaw)*
- [ ] **Nextcloud Talk** - Self-hosted Nextcloud chat. *(OpenClaw)*
- [ ] **Zalo** - Zalo Bot API + personal account via QR login. *(OpenClaw)*
- [ ] **Tlon Messenger** - P2P ownership-first chat. *(OpenClaw)*
- [ ] **WebChat** - Browser-based UI for direct agent interaction. *(OpenClaw)*

## 18. Search Engine

- [ ] **SearXNG Integration** - Self-hosted meta-search using top 10 providers. Ships with framework. *(Original - notes.md)*
- [ ] **Go-native Search** - Research: Can we build a Go version of SearXNG? *(Original - notes.md)*

## 19. Security

- [ ] **Outbound PII Scrubbing** - Scramble ALL private info before it leaves the agent: API keys, tokens, addresses, SSNs, phone numbers, credit cards. Nothing leaks to providers. *(Original - notes.md)*
- [x] **Shell Safety** - Blocked commands, protected paths. *(Original)*
- [x] **PII Detection** - Guardrails middleware scans for SSN, credit cards. *(iTaK Gateway)*
- [ ] **Dashboard Login** - Username/password authentication for iTaK Dashboard and API access. Bcrypt password hashing. *(Original)*
- [ ] **2FA / Two-Factor Auth** - TOTP-based second factor using authenticator apps (Google Authenticator, Authy, Microsoft Authenticator). Go-native via `pquerna/otp`. *(Original)*
- [ ] **QR Code Pairing** - On first 2FA setup, generate a QR code the user scans with their authenticator app. Go-native via `skip2/go-qrcode`. *(Original)*
- [ ] **Session Management** - JWT or cookie-based sessions with configurable expiry. Auto-logout on inactivity. *(Original)*
- [ ] **Device Trust** - Remember trusted devices so 2FA isn't needed every login. Revoke trusted devices from dashboard. *(Original)*
- [ ] **API Key Encryption** - Store keys encrypted, not plaintext. *(Best practice)*
- [ ] **Database Encryption** - Encrypt all data at rest. *(Original - notes.md)*
- [ ] **Private Info Manager** - Custom section for secrets stored in .env. Dashboard + CLI can set/view. *(Original - notes.md)*
- [ ] **Security Audit** - Review framework. Study OpenClaw security patches. *(OpenClaw)*
- [ ] **Zero Data Retention** - Never store user data beyond session. Never train on user data. More secure than using OpenAI/Gemini directly. *(ClickUp Super Agents)*
- [ ] **Agentic User Security** - Permission model: implicit access, explicit access, custom permissions. Agents inherit user permissions. Full audit trail of every action. *(ClickUp Super Agents)*

## iTaK Auth

### iTaK Auth (Enterprise Identity & Access Management)

Dedicated auth service for the GO* ecosystem. Fork from **Ory Hydra** (OAuth2/OIDC server) + **Ory Kratos** (identity management) - both Go-native, both used by Fortune 500 companies. All GO* services authenticate through iTaK Auth.

**Core Identity**
- [ ] **OAuth2 / OIDC Server** - Industry-standard auth. Issues access tokens, refresh tokens, ID tokens. Any corporate client already speaks this protocol. *(Ory Hydra fork)*
- [ ] **User Management** - Registration, login, password reset, email verification, profile management. *(Ory Kratos fork)*
- [ ] **Single Sign-On (SSO)** - Log in once, access iTaK Torch, iTaK Media, iTaK Dashboard, iTaK Gateway, iTaK Forge. One session across all services. *(Ory Hydra)*
- [ ] **Service Accounts** - Machine-to-machine auth via client credentials grant. For iTaK-to-iTaK service communication. *(OAuth2 standard)*

**Security**
- [ ] **MFA / 2FA** - TOTP (Google Authenticator), WebAuthn/Passkeys (hardware keys, fingerprint), SMS backup. *(Ory Kratos)*
- [ ] **Session Management** - Short-lived access tokens (15 min), long-lived refresh tokens (7 days). Auto-revoke on suspicious activity. *(OAuth2 standard)*
- [ ] **Audit Trail** - Every auth event logged: logins, token grants, permission changes, failed attempts. Exportable to SIEM. *(Original)*

**Access Control**
- [ ] **RBAC (Role-Based Access)** - Roles: `admin`, `operator`, `viewer`, `service`. Per-service permission scopes. *(Original)*
- [ ] **API Key Management** - Generate/revoke scoped API keys per service or external client. Rate limiting per key. *(Original)*
- [ ] **Auth Middleware (`pkg/auth/`)** - Shared Go package that any GO* service imports. One-line auth enforcement. *(Original)*

**Federation (Connect to Corporate Systems)**
- [ ] **External IdP Federation** - Connect to Okta, Azure AD, Google Workspace, Auth0, OneLogin via OIDC/SAML. *(Ory Hydra)*
- [ ] **LDAP / Active Directory** - Enterprise directory integration for on-prem corporate environments. *(Dex patterns)*
- [ ] **SCIM Provisioning** - Auto-sync users from corporate directory. When someone joins/leaves the company, iTaK Auth updates automatically. *(Enterprise standard)*

**Deployment Modes**
- [ ] **Embedded (Default)** - iTaK Auth runs as a library inside iTaK Agent. Zero extra processes. *(Original)*
- [ ] **Standalone Service** - Run iTaK Auth as its own process for multi-machine setups. *(Original)*
- [ ] **Auto-Pairing (Local)** - Services on same machine auto-discover via shared local secret. Zero config. *(Original)*
- [ ] **Remote Pairing** - Cross-machine: `itakagent pair <service-url>` exchanges keys via one-time code. *(Original)*

**iTaK Auth Mobile App (Authenticator)**
- [ ] **Cross-Platform App** - Built with Fyne (Go GUI toolkit). Single codebase compiles to iOS, Android, Windows, macOS. *(Original, Fyne)*
- [ ] **TOTP Code Generator** - 6-digit rotating codes (30-sec). Works offline. Compatible with Google Authenticator standard. *(Original)*
- [ ] **Push Approve/Deny** - "Login attempt from Windows PC in [City]. [Approve] [Deny]" notifications. *(Original)*
- [ ] **Biometric Unlock** - Fingerprint/Face ID to approve auth requests. *(Original)*
- [ ] **QR Pairing** - Scan QR code from iTaK Dashboard to pair phone with iTaK Auth server. *(Original)*
- [ ] **Multi-Account** - Manage multiple iTaK Agent instances from one phone app. *(Original)*
- [ ] **PWA Fallback** - Progressive Web App version for browsers. No app store install needed. Offline TOTP via Service Worker. *(Original)*

## 20. Integrations & Skills

- [ ] **Figma** - Design-to-code pipeline. *(Original)*
- [ ] **Unreal Engine** - Game dev via MCP. *(Original)*
- [ ] **n8n Workflow Calls** - Agents can trigger n8n workflows via webhooks/WebSocket. *(Original - notes.md)*
- [ ] **No-Code Platform Integration** - Zapier, Make, IFTTT, Power Automate. *(Original - notes.md)*
- [ ] **Draw.io / Diagrams.net** - Architecture diagrams, flowcharts, ERDs. Self-hostable via Docker. *(Original - tasks.md)*
  - Repos: https://github.com/jgraph/drawio, https://github.com/jgraph/docker-drawio, https://github.com/jgraph/drawio-diagrams
- [ ] **Obsidian** - Knowledge graph notes integration. *(OpenClaw)*
- [ ] **Notion** - Workspace and database automation. *(OpenClaw)*
- [ ] **Trello** - Kanban board integration. *(OpenClaw)*
- [ ] **Home Assistant** - Smart home automation hub. *(OpenClaw)*
- [ ] **Spotify** - Music playback control for ambient agent environments. *(OpenClaw)*
- [ ] **Sonos** - Multi-room audio control. *(OpenClaw)*
- [ ] **Shazam** - Song recognition. *(OpenClaw)*
- [ ] **Weather** - Forecasts and conditions. Location-aware alerts. *(OpenClaw)*
- [ ] **Camera / Webcam** - Photo and video capture from connected cameras/webcams. Agent can see the user and their environment via webcam feed, processed through vision models (moondream2, qwen3-vl) via iTaK Torch + iTaK Media. *(OpenClaw, Original)*
- [ ] **GIF Search** - Find and send the perfect GIF. Integrates into chat and social media agents. *(OpenClaw)*
- [ ] **Peekaboo** - Quick screen capture and share. Lightweight alternative to full iTaK Vision for simple screenshots. *(OpenClaw)*
- [ ] **Genspark Integration** - [genspark.ai](https://www.genspark.ai/) AI-powered search engine. Research their agent-to-search patterns for deep web research capabilities. *(Original)*

## 21. Platform Support

- [x] **Windows** *(Primary)*
- [x] **Linux** *(Cross-compile)*
- [ ] **macOS** - Menu bar app + Voice Wake. *(Go cross-compile, OpenClaw)*
- [ ] **iOS** - Canvas, camera, Voice Wake companion app. *(OpenClaw)*
- [ ] **Android** - Canvas, camera, screen companion app. *(OpenClaw)*
- [ ] **Chromebook** *(Linux layer)*
- [ ] **Low-spec Hardware** - Target: old i7 mini PC, 16GB RAM (Dell OptiPlex 7060). *(Original)*

## iTaK IDE

### iTaK IDE (Standalone AI-Native Development Environment)

Full standalone IDE with built-in AI agent orchestration. Same architecture as VS Code, Cursor, and Windsurf: Electron frontend (React/TypeScript) with iTaK Go engine as the backend. Ships as a single desktop app with everything baked in.

**Architecture:**
```
┌─────────────────────────────────────┐
│           Electron Shell            │
│  ┌───────────────────────────────┐  │
│  │     React/TypeScript UI       │  │
│  │  - Chat panel (agent chat)    │  │
│  │  - Monaco editor (code)       │  │
│  │  - File explorer              │  │
│  │  - Terminal panel             │  │
│  │  - Settings/config            │  │
│  └──────────┬────────────────────┘  │
│             │ IPC / WebSocket        │
└─────────────┼───────────────────────┘
              │
┌─────────────┴───────────────────────┐
│         Go Backend (iTaK Core)       │
│  - iTaK Torch inference engine       │
│  - Agent orchestration (AgentZero)   │
│  - Tool execution (file, terminal)   │
│  - LSP bridge (gopls, pyright, etc)  │
│  - Memory/context management         │
│  - Model management (download, swap) │
└─────────────────────────────────────┘
```

### Editor Core (Monaco)
- [ ] **Monaco Editor Integration** - VS Code's editor component. Syntax highlighting, IntelliSense, minimap, multi-cursor, diff view, bracket matching, code folding. *(VS Code/Monaco)*
- [ ] **Multi-Tab Editing** - Open multiple files in tabs. Split panes, side-by-side diff. *(VS Code pattern)*
- [ ] **File Explorer** - Tree view of project files with search, create, rename, delete. Git status indicators. *(VS Code pattern)*
- [ ] **Integrated Terminal** - Full terminal emulator. Multiple terminals, split view. Go backend spawns shells. *(VS Code pattern)*
- [ ] **Command Palette** - Ctrl+Shift+P quick command access. Fuzzy search across all commands, files, and settings. *(VS Code pattern)*
- [ ] **Theme System** - Dark/light mode, custom color themes. Ship with iTaK branded dark theme. *(VS Code pattern)*
- [ ] **Keybinding System** - Customizable keybindings. VS Code, Vim, Emacs presets. *(VS Code pattern)*

### Built-in Agent Chat
- [ ] **Agent Chat Panel** - Side panel for chatting with AgentZero and all agent personas. Streaming markdown responses. *(Cursor/Windsurf pattern)*
- [ ] **Inline Code Actions** - Right-click code to "Explain", "Refactor", "Add Tests", "Find Bugs". Agent responds in chat with diffs. *(Cursor pattern)*
- [ ] **@ Mentions** - @agent to talk to specific agents. @coder, @researcher, @devops. *(Copilot Chat pattern)*
- [ ] **Context Awareness** - Agent sees open file, cursor position, selection, terminal output, git diff. No manual copy-paste. *(Cursor/Windsurf pattern)*
- [ ] **Apply Diffs** - Agent suggests code changes as diffs. One-click accept/reject per hunk. *(Cursor pattern)*
- [ ] **Multi-File Edits** - Agent can edit multiple files in one response. Preview all changes before applying. *(Windsurf Cascade pattern)*

### Go Backend Integration
- [ ] **Local WebSocket/IPC Bridge** - Electron communicates with Go backend via local WebSocket or Node IPC. Low latency, fully local. *(Original)*
- [ ] **iTaK Torch Embedded** - Inference engine runs as part of the Go backend process. No separate server needed. Model management from IDE settings. *(Original)*
- [ ] **LSP Bridge** - Go backend manages Language Server Protocol connections. gopls (Go), pyright (Python), typescript-language-server (TS/JS), rust-analyzer (Rust). *(VS Code pattern)*
- [ ] **Tool Execution Engine** - File operations, terminal commands, browser automation all executed by Go backend. Same tool system as iTaK Agent CLI. *(Original)*
- [ ] **Memory System** - Agent memory, knowledge graph, and context management via Go backend. Persists across IDE sessions. *(Original)*
- [ ] **Model Management UI** - Settings page for downloading, swapping, and configuring local models. Hardware auto-detection. *(Original)*

### IDE-Specific Features
- [ ] **Project Dashboard** - Overview of project: files, agents active, recent changes, health status. *(Original)*
- [ ] **Kanban Integration** - Built-in Kanban board for agent tasks. Watch AI build your project in real-time. *(Original, Kanban feature)*
- [ ] **Git Integration** - Built-in Git panel. Stage, commit, push, pull, branch, merge. Visual diff. *(VS Code pattern)*
- [ ] **Extension System** - iTaK Hub extensions installable directly in the IDE. Language support, themes, tools. *(VS Code Marketplace pattern)*
- [ ] **Settings Sync** - Sync IDE settings, keybindings, and extensions across machines. *(VS Code pattern)*
- [ ] **iTaK Dashboard Embed** - Full iTaK Dashboard accessible as a tab within the IDE. Agent monitoring, cost tracking, knowledge graph. *(Original)*

### Phased Implementation
| Phase | Deliverable | Description |
|-------|-------------|-------------|
| **Phase 1** | VS Code Extension | Chat Participant API. `@itak` in VS Code Copilot Chat. Validates the UX. |
| **Phase 2** | Electron MVP | Monaco editor + chat panel + Go backend. Basic file editing + agent chat. |
| **Phase 3** | Full IDE | Terminal, Git, Kanban, extensions, model management. Production-ready. |

### References
| Project | URL | Relevance |
|---------|-----|----------|
| VS Code (OSS) | https://github.com/microsoft/vscode | Architecture reference, Monaco editor, extension system |
| Monaco Editor | https://github.com/microsoft/monaco-editor | Code editor component |
| Electron | https://github.com/electron/electron | Desktop shell |
| Cursor | https://cursor.com | AI-native IDE UX patterns |
| Windsurf | https://windsurf.com | Cascade multi-file edit patterns |
| Zed | https://github.com/zed-industries/zed | Performance-first editor (Rust, study perf patterns) |

## iTaK Torch

### 22. iTaK Torch (Go-Native Inference Engine)

Custom Go-native LLM inference runtime. No dependency on Ollama or any external tool. iTaK Agent loads and runs models by itself.

- [ ] **GGUF Model Loader** - Load quantized GGUF models directly via CGo bindings to `llama.cpp`. No Ollama required. *(Original)*
- [ ] **HuggingFace Model Pull** - Download models directly from HuggingFace Hub. Browse, search, and pull GGUF files by repo name. *(Original)*
- [ ] **Ollama Registry Pull** - Also pull models from the Ollama registry if users prefer that catalog. Best of both worlds. *(Original)*
- [ ] **Local File Support** - Point iTaK Torch at any local `.gguf` file and it just works. *(Original)*
- [ ] **Model Cache** - Downloaded models stored in `~/.itaktorch/models/`. No re-download. Shared across all iTaK Agent instances on the machine. *(Original)*
- [ ] **Runtime Model Swapping** - Hot-swap models at runtime. Boss says "switch coder to qwen2.5-coder" and it loads in seconds. *(Original)*
- [ ] **Multi-Model Concurrent** - Run multiple models simultaneously (embedding + chat + coding). Memory-aware: only loads what fits in RAM. *(Original)*
- [ ] **CPU Optimized** - AVX2/AVX-512 auto-detection for maximum CPU inference speed. No GPU required. *(Original)*
- [ ] **GPU Acceleration (Optional)** - CUDA, ROCm, Metal support for users who have a GPU. Auto-detect and use if available. *(Original)*
- [ ] **OpenAI-Compatible API** - iTaK Torch exposes a local `/v1/chat/completions` endpoint so iTaK Gateway and all agents can talk to it like any cloud provider. *(Original)*
- [ ] **Quantization on Download** - Auto-quantize models to Q4_K_M during download if the user's hardware needs it. *(Original)*
- [ ] **Inference Metrics** - Tokens/second, memory usage, model load time. Feed into iTaK Dashboard. *(Original)*
- [ ] **Request Queue** - Thread-safe FIFO queue for inference requests. Multiple users/agents pushing prompts get queued and processed in order. Priority lanes for system-critical requests vs user chat. Backpressure: reject new requests when queue exceeds configurable depth. Queue depth visible in `/health` and iTaK Dashboard. *(Original)*
- [ ] **Research**: Study `go-skynet/LocalAI`, `mudler/go-llama.cpp`, `ggml-org/llama.cpp` CGo patterns, and `ollama/ollama` internals for architecture inspiration. *(Research)*

### Pure Go Backend (GOTensor)
- [ ] **GOTensor Engine** - Pure Go tensor engine in `pkg/torch/native/`. AVX-512 optimized matmul, convolutions, packed GEMM. Zero CGo deps. Forked and rebranded from GoMLX's `backends/simplego/`. *(GOTensor)*
- [ ] **Native Inference Engine** - Run tiny models (<1B params) entirely in pure Go. No shared libraries needed at all. *(GOTensor, Original)*
- [ ] **Training & Fine-Tuning** - Train and fine-tune models locally using GOTensor's XLA GPU backend. Agents can autonomously create custom models from user data. Dashboard section for training jobs, loss curves, and model management. *(GOTensor, Original)*

### LLM Provider System (from LangChainGo)
- [ ] **Unified Model Interface** - Fork LangChainGo's `Model` interface with `GenerateContent()` and `ReasoningModel` support. *(LangChainGo)*
- [ ] **Cloud Provider Clients** - Fork provider implementations: OpenAI, Anthropic, Ollama, Gemini, Mistral, HuggingFace, Cloudflare, Cohere. *(LangChainGo)*
- [ ] **Response Cache Layer** - Wrap any provider with disk/memory cache. Don't re-call identical prompts. *(LangChainGo)*
- [ ] **Error Mapping** - Maps provider-specific errors to common error types. *(LangChainGo)*
- [ ] **Token Counting** - Count tokens without loading a model. *(LangChainGo)*

### Agent Tools (from LangChainGo)
- [ ] **DuckDuckGo Search** - Free web search, no API key. *(LangChainGo)*
- [ ] **Wikipedia Lookup** - Article search and retrieval. *(LangChainGo)*
- [ ] **Web Scraper** - Extract content from web pages. *(LangChainGo)*
- [ ] **SQL Database Tool** - Execute SQL queries against any database. *(LangChainGo)*
- [ ] **Calculator** - Math expression evaluation. *(LangChainGo)*
- [ ] **Perplexity Search** - AI-powered search. *(LangChainGo)*

### Agent Patterns (from Eino)
- [ ] **Stream Processing** - Auto-manage token streams between agent components. Concatenating, boxing, merging streams. *(Eino/ByteDance)*
- [ ] **Interrupt/Resume** - Pause agent execution for human approval, resume from checkpoint. State persistence. *(Eino/ByteDance)*
- [ ] **Callback Aspects** - Inject logging/tracing/metrics at OnStart/OnEnd/OnError hooks across all components. *(Eino/ByteDance)*

### ONNX Support (from Hugot)
- [ ] **ONNX Model Runner** - Run HuggingFace ONNX models for embeddings, classification, text generation. *(Hugot)*
- [ ] **Local Embedding Pipeline** - Generate embeddings locally without API calls using ONNX models. *(Hugot)*

### Caching & Performance (from Ecosystem Research)
- [ ] **Ristretto Cache** - High-performance Go cache (SampledLFU + TinyLFU, 26.9k dependents). Use for prompt/embedding/tokenization caching. `dgraph-io/ristretto`. *(Research Sweep)*
- [ ] **Go 1.26 SIMD** - Experimental `simd/archsimd` package for 128/256/512-bit vector ops. Potential for embedding math acceleration. `GOEXPERIMENT=simd`. *(Go 1.26)*
- [ ] **Go 1.26 runtime/secret** - Secure register/stack erasure for API key handling. `GOEXPERIMENT=runtimesecret`. *(Go 1.26)*
- [ ] **Go 1.26 Green Tea GC** - Now default. 10-40% GC overhead reduction. Vector-accelerated small object scanning. *(Go 1.26)*
- [ ] **Binary Size Optimization** - Use `goda` (dep graph) and `go-size-analyzer` (`gsa --web`) to audit and shrink iTaK Torch binary. Datadog achieved 77% reduction. *(Research Sweep)*

### Code Analysis Tools (from Ecosystem Research)
- [ ] **gotreesitter** - Pure Go tree-sitter runtime. No CGo. Parser, lexer, query engine, incremental reparsing. Matches our purego philosophy. Use as code analysis skill for agents. `odvcencio/gotreesitter`. *(Research Sweep)*

### Agent State Management (from Ecosystem Research)
- [ ] **Stateless FSM** - Go finite state machine with hierarchical states, entry/exit events, guard clauses, external state storage, DOT graph export. Use for agent state (idle -> thinking -> tool_calling -> responding). `qmuntal/stateless`. *(Research Sweep)*
- [ ] **DBOS Durable Execution** - Extends `context.Context` into `durable.Context` for crash-recoverable workflow checkpointing. Type-safe `RunWorkflow`/`RunAsStep`. Use for long-running agent pipelines. *(Research Sweep)*

### Tool Patterns (from go-llm / gollm Research)
- [ ] **KeyValueStore Tool** - Reduces context by storing long content by-reference. Agent uses keys instead of repeating data. *(go-llm pattern)*
- [ ] **JSONAutoFixer Meta-Tool** - When tool args are malformed JSON, auto-fix via separate LLM call before retrying. *(go-llm pattern)*
- [ ] **GenericAgentTool** - Lets one agent spawn another with dynamic task and tools. *(go-llm pattern)*
- [ ] **Prompt Optimizer** - Auto-improve prompts via LLM feedback loop. Chain-of-thought pre-built functions. Model comparison (run same prompt against N models). *(gollm pattern)*
- [ ] **Structured JSON Validation** - Validate LLM output against JSON schema at the framework level. *(gollm pattern)*

### Security Hardening (from OpenClaw Audit Cross-Check)
- [ ] **Path Traversal Validation** - Validate `--model` and `--mmproj` paths (no `..` traversal, must end in `.gguf`). *(OpenClaw Security Audit)*
- [ ] **Prompt Sanitization** - Strip potential PII from prompts before logging. *(OpenClaw Security Audit)*
- [ ] **API Rate Limiting** - Rate-limit inference endpoints to prevent abuse. *(OpenClaw Security Audit)*

## iTaK Media

### 23. iTaK Media (Go-Native Media Downloader & Transcriber)

Go-native alternative to yt-dlp targeting the top 15 social media platforms. Single binary, no Python dependency. Fork base from `kkdai/youtube` (Go YouTube library) and `horiagug/youtube-transcript-api-go`.

### Core Engine
- [ ] **Platform Abstraction** - Unified `Extractor` interface per platform. Each implements: resolve URL, get metadata, get streams, get subtitles. *(Original)*
- [ ] **Stream Downloader** - HTTP, HLS (m3u8), DASH stream downloading with resume support. Goroutine-based parallel chunk downloads. *(Original)*
- [ ] **Subtitle Extractor** - Pull auto-generated and manual subtitles/captions. Multi-language. VTT/SRT/JSON output. *(Original)*
- [ ] **Audio Extractor** - Download audio-only streams for Whisper fallback transcription. *(Original)*
- [ ] **Whisper Integration** - For videos without subtitles: download audio, transcribe via local Whisper model or Whisper API. *(Original)*
- [ ] **FFmpeg Bridge** - Merge audio+video streams, convert formats. Optional dep for advanced use. *(Original)*
- [ ] **Clean Transcript Output** - Strip timestamps, dedupe repeated lines, format as readable text or structured JSON. *(Original)*

### Platform Extractors (Top 15)
| # | Platform | Auth | Notes |
|---|----------|------|-------|
| 1 | **YouTube** | Cookie/OAuth | Fork `kkdai/youtube`. Subtitles via transcript API |
| 2 | **TikTok** | Cookie | Video + captions extraction |
| 3 | **Instagram** | Cookie/Session | Reels, Stories, IGTV. Login required for private |
| 4 | **X/Twitter** | Cookie/Bearer | Spaces audio, video tweets |
| 5 | **Reddit** | API/Cookie | v.redd.it video extraction |
| 6 | **Facebook** | Cookie | **Private video support via session cookies** |
| 7 | **Twitch** | API | VODs, clips, live streams |
| 8 | **Vimeo** | API/Cookie | Public + private (password-protected) |
| 9 | **LinkedIn** | Cookie | Learning videos, feed videos |
| 10 | **Threads** | Cookie | Meta's Threads platform |
| 11 | **Bluesky** | API | AT Protocol video |
| 12 | **SoundCloud** | API | Audio tracks, podcasts |
| 13 | **Spotify** | Cookie | Podcast episodes (audio) |
| 14 | **Rumble** | Public | Video extraction |
| 15 | **Kick** | API/Cookie | VODs and clips |

### Authentication / Private Content
- [ ] **Cookie Import** - Import browser cookies for authenticated access. Read from Chrome/Firefox/Edge cookie stores. *(yt-dlp pattern)*
- [ ] **Session Token Auth** - Store and reuse session tokens per platform. Encrypted at rest. *(Original)*
- [ ] **Browser Profile Bridge** - Use iTaK Browser's dedicated profile for authenticated downloads. Handles 2FA, CAPTCHA. *(Original)*
- [ ] **OAuth Flows** - Where platforms support it (YouTube, Twitch), use proper OAuth for stable access. *(Original)*
- [ ] **Private Video Support** - Facebook private videos, Instagram private accounts, unlisted YouTube via auth cookies. *(Original)*

### CLI & Library
- [ ] **Standalone CLI** - `gomedia download URL`, `gomedia transcript URL`, `gomedia info URL`. *(Original)*
- [ ] **Go Library** - Import as `pkg/media/` in iTaK Agent. Agents call it directly, no shell-out. *(Original)*
- [ ] **yt-dlp Fallback** - If iTaK Media doesn't support a platform, shell out to yt-dlp if installed. Graceful degradation. *(Original)*

## 24. API Server

- [ ] **REST API** - Standalone HTTP API for external apps to interact with iTaK Agent. Send tasks, query agents, get results. *(Recommendation)*
- [ ] **gRPC API** - High-performance binary protocol for Go-to-Go and mobile app integration. *(Recommendation)*
- [ ] **API Key Auth** - Secure API access with generated API keys. Rate limiting per key. *(Recommendation)*
- [ ] **Webhook Callbacks** - Register webhook URLs. iTaK Agent calls back when tasks complete. *(Recommendation)*

## 25. Observability & Export

- [ ] **Log Export** - Export structured logs to Grafana/Loki, ELK stack, or plain JSON files. *(Recommendation)*
- [ ] **Trace Export** - OpenTelemetry-compatible traces. Full request lifecycle from user input to agent response. *(Recommendation)*
- [ ] **Metrics Export** - Prometheus-compatible metrics endpoint. Token usage, response times, agent utilization. *(Recommendation)*

## 26. Missing Infrastructure (Gap Analysis)

Features identified by asking: "If everything above was built, what would still be missing?"

### Installation & Distribution
- [ ] **One-Line Installer** - `curl -sSL install.itakagent.dev | sh` for Linux/Mac. PowerShell equivalent for Windows. Detects OS/arch, downloads binary, sets PATH. *(Recommendation)*
- [ ] **Prerequisite Auto-Installer** - On first run, scan for all required dependencies (Go, Git, CMake, WSL, ffmpeg, etc.). Missing items are auto-installed with user confirmation. Windows: uses winget/choco. Linux: apt/dnf. Mac: brew. *(Original)*
- [ ] **WSL-First on Windows** - On Windows, default to running inside WSL (Ubuntu). Installer checks for WSL, installs it if missing (`wsl --install`), then sets up iTaK Agent inside WSL. User can opt-out to native Windows mode via `--native` flag. WSL gives better Go/Linux compat, avoids firewall popups, and matches production Linux servers. *(Original)*
- [ ] **Auto-Updater** - Background check for new versions. Notifies user, downloads, restarts. Rollback if update fails. *(Recommendation)*
- [ ] **Homebrew / Winget / Snap** - Package manager distribution for easy install and updates. *(Recommendation)*
- [ ] **Docker Image** - Official `itakagent/itakagent` Docker image on GHCR. Multi-arch (amd64, arm64). *(Recommendation)*

### CLI Experience
- [ ] **Interactive Setup Wizard** - First-run TUI wizard: choose providers, paste API keys, select hardware tier, pick default model. Beautiful terminal UI via `charmbracelet/bubbletea`. *(Recommendation)*
- [ ] **Shell Completions** - Bash, Zsh, Fish, PowerShell auto-completions for all CLI commands. *(Recommendation)*
- [ ] **CLI Themes** - Configurable terminal colors, emoji usage, output density (compact/verbose/json). *(Recommendation)*

### Documentation & Help
- [ ] **Built-in Docs Server** - `itakagent docs` starts a local docs site. API reference, tutorials, architecture guides. *(Recommendation)*
- [ ] **`itakagent help <topic>`** - Context-aware help. `itakagent help agents` shows all agent types, `itakagent help itaktorch` shows model management. *(Recommendation)*
- [ ] **Example Projects** - Curated example projects showing real-world agent setups: "Build a blog with agents", "Automate social media", "Run a research pipeline". *(Recommendation)*
- [ ] **Video Tutorials** - Agent-generated video walkthroughs (using iTaK Media + iTaK Vision to record and narrate). *(Recommendation)*

### Testing & Quality
- [ ] **Agent Test Framework** - Write tests for agent behaviors: "Given this input, agent should use this tool and produce this output." CI-friendly. *(Recommendation)*
- [ ] **Simulation Mode** - Run agents against mock LLMs and mock tools. Test workflows without burning API tokens. *(Recommendation)*
- [ ] **Regression Tests** - Track agent quality over time. Same prompt should produce same-quality output after updates. *(Recommendation)*

### Data Management
- [ ] **Import/Export Wizard** - Migrate from other agent frameworks (Agent Zero, OpenClaw, LangChain). Import configs, memories, prompts. *(Recommendation)*
- [ ] **Data Portability** - All user data exportable as standard formats (JSON, CSV, SQLite). No vendor lock-in. *(Recommendation)*
- [ ] **Multi-Machine Sync** - Sync knowledge graphs, memories, and configs across multiple iTaK Agent instances. CRDTs or last-write-wins. *(Recommendation)*

### Notifications & Alerts
- [ ] **Notification Center** - Unified notification hub in iTaK Dashboard. Agent completions, errors, approvals needed, health alerts. *(Recommendation)*
- [ ] **Alert Rules** - Configurable alerts: "Notify me on Discord when spending exceeds $5/day", "Email me when a deploy fails". *(Recommendation)*
- [ ] **Quiet Hours** - Do Not Disturb mode. Batch non-urgent notifications. *(Recommendation)*

### Resilience & Recovery
- [ ] **Graceful Degradation** - If iTaK Torch dies, auto-fallback to cloud providers. If internet dies, auto-fallback to local models. No user intervention. *(Recommendation)*
- [ ] **Task Persistence** - Running tasks survive process restarts. Resume from last checkpoint on reboot. *(Recommendation)*
- [ ] **Disaster Recovery** - Full backup/restore of entire iTaK Agent state. Point-in-time recovery. *(Recommendation)*
- [ ] **Migration CLI** - `itakagent migrate` to move entire setup between machines. *(Recommendation)*

### Rate Limiting & Abuse Prevention
- [ ] **Per-User Rate Limits** - In multi-user mode, prevent one user from consuming all resources. *(Recommendation)*
- [ ] **Agent Spending Caps** - Per-agent daily/monthly spending limits. Auto-pause agent when exceeded. *(Recommendation)*
- [ ] **Resource Quotas** - Limit CPU/RAM/disk per user or project in multi-tenant setups. *(Recommendation)*

### Internationalization
- [ ] **Multi-Language UI** - iTaK Dashboard in English, Spanish, Portuguese, French, German, Japanese, Chinese, Korean, Arabic. *(Recommendation)*
- [ ] **Locale-Aware Agents** - Agents respect user's date/time/currency formats. *(Recommendation)*

### Agent-to-Agent Communication
- [ ] **A2A Protocol** - Google's Agent-to-Agent protocol for cross-framework agent communication. iTaK Agent agents can talk to agents from other platforms. *(Google A2A)*
- [ ] **Agent Discovery** - Publish agent capabilities so external agents can discover and invoke them. *(Google A2A)*

### Billing & Licensing (for SaaS/Enterprise)
- [ ] **Usage Metering** - Track per-user, per-agent, per-project resource consumption for billing. *(Recommendation)*
- [ ] **License Key System** - For commercial distribution. Free tier, Pro tier, Enterprise tier. *(Recommendation)*
- [ ] **White-Label** - Remove iTaK Agent branding. Companies deploy under their own brand. *(Recommendation)*

### Telemetry (Opt-in)
- [ ] **Anonymous Usage Stats** - Opt-in telemetry: which features are used, error rates, performance metrics. For improving iTaK Agent. *(Recommendation)*
- [ ] **Crash Reporting** - Automatic crash reports with stack traces. Opt-in only. *(Recommendation)*

## 27. Autonomy Engine (GOPilot)

The core differentiator. iTaK Agent should run unattended for hours, fix its own problems, and only bother the user when it genuinely can't proceed. This section covers everything needed to make that real.

### Self-Correction & Error Recovery
- [x] **7-Step Escalation Chain** - On failure, the agent walks this ladder automatically before EVER asking the user: `pkg/agent/autonomy.go`
  1. **Retry** - Same model, same approach, 2 attempts
  2. **Check Error Database** - Has this error been seen before? Apply known fix
  3. **Escalate Model** - Try a bigger/smarter model on the same task
  4. **Try Different Approach** - Rephrase the task, use alternative tools
  5. **Research Online** - Search the web for the error message and apply what it finds *(see below)*
  6. **Self-Debug** - Dump full state to a fresh agent for diagnosis
  7. **Ask User** - Only after ALL above steps fail. Include what was tried and what was found online
- [ ] **Research Before Asking** - When an agent encounters an unknown error, it MUST search the web before asking the user. Uses SearXNG to search Stack Overflow, GitHub Issues, official docs, and forums. Extracts fix candidates, attempts them, and only escalates to user if no fix works. *(Original)*
- [ ] **Error-to-Search Pipeline** - Auto-extract the error message, strip file paths/line numbers, search for the core error pattern. Parse top 5 results for code fixes. *(Original)*
- [ ] **GitHub Issues Search** - Search the specific library/tool's GitHub Issues for the exact error. Often the fix is already posted. *(Original)*
- [ ] **Official Docs Lookup** - Check official documentation for the tool/library/API that threw the error. Use Context7 or direct docs scraping. *(Original)*
- [ ] **Stack Overflow Extraction** - Parse Stack Overflow answers, extract code blocks from accepted/highest-voted answers, adapt to current context. *(Original)*
- [ ] **Fix Attempt Loop** - Try each candidate fix from web research. If fix A fails, try fix B. Log what was tried and what worked for the Error Pattern Database. *(Original)*
- [x] **Output Validation Loop** - After every tool call, validate the result. Did the file actually save? Did the API return 200? Did the code compile? Re-run if not. *(Original)*
- [ ] **Code Compile Check** - After writing code, auto-compile/lint. If errors, fix them in a loop (max 5 attempts) before reporting success. *(Original)*
- [ ] **Rollback on Failure** - If an agent breaks something (corrupted file, crashed service), auto-rollback to last known good state via git/snapshot. *(Original)*
- [x] **Error Pattern Database** - Structured database of error signatures to fix patterns. Error X seen before = instant fix from memory. Grows over time. Includes fixes found via web research. `pkg/agent/doctor.go` *(Original)*
- [ ] **Self-Debug Mode** - When agent gets stuck, it dumps its own state (last N messages, tool results, memory) and feeds it to a fresh agent instance for diagnosis. *(Original)*

### Confidence & Escalation
- [ ] **Confidence Scoring** - Every agent response includes a self-rated confidence score (0-100). Low confidence = auto-escalate to bigger model or human review. *(Original)*
- [ ] **Uncertainty Detection** - Detect hedging language ("I think", "maybe", "probably") and auto-flag for verification. *(Original)*
- [ ] **Human-in-the-Loop Threshold** - Configurable confidence threshold. Below it, pause and ask user. Default: 30%. Power users can set to 0% (full autopilot). *(Original)*
- [ ] **Escalation Routing** - Route difficult tasks to stronger models automatically. Simple tasks stay on cheap/fast models. Based on task complexity classification. *(Original)*

### Process Watchdogs
- [ ] **Watchdog Process** - Separate lightweight process monitors iTaK Agent. Detects freezes, infinite loops, memory leaks. Auto-restart with state recovery. *(Original)*
- [ ] **Deadlock Detection** - Detect when agents are waiting on each other circularly. Auto-break the cycle by killing the youngest task. *(Original)*
- [ ] **Stale Task Cleanup** - Tasks running beyond expected time with no progress get auto-killed. Configurable timeout per task type. *(Original)*
- [ ] **Memory Leak Detection** - Monitor per-agent RAM usage over time. If growing without bound, restart that agent's process. *(Original)*
- [ ] **Zombie Agent Cleanup** - Detect agents that are alive but not doing anything. Terminate and reclaim resources. *(Original)*

### Dependency & Environment Healing
- [ ] **Dependency Doctor** - On startup and on error, check all external deps (git, ffmpeg, node, python, go). Auto-install missing ones. *(Original)*
- [ ] **Config Validator** - On startup, validate entire itakagent.yaml. Fix common misconfigs automatically (wrong paths, invalid ports, missing fields). *(Original)*
- [ ] **Port Conflict Resolution** - If a port is in use, auto-pick next available. No manual intervention. *(Original)*
- [ ] **Disk Space Monitor** - Alert and auto-clean (temp files, old logs, cached models) if disk space is low. *(Original)*
- [ ] **Network Health Check** - Continuously verify internet connectivity and API reachability. Auto-switch to offline mode if connection drops, switch back when restored. *(Original)*

### Self-Improvement
- [ ] **Self-Benchmarking** - Periodically run standardized tasks and measure quality/speed. Detect quality degradation after model swaps or config changes. *(Original)*
- [ ] **Prompt Effectiveness Tracking** - Track which prompts produce good results per model. Auto-retire underperforming prompts. *(Original)*
- [ ] **Task Success Rate** - Track completion rates per agent and task type. Flag agents with declining success for retraining or prompt revision. *(Original)*
- [ ] **Learning from Corrections** - When user corrects agent output, store the correction as a training example. Future similar tasks reference corrections. *(Original)*

### Autonomy Levels
User-configurable autonomy from fully supervised to fully autonomous:

| Level | Name | Behavior |
|-------|------|----------|
| 0 | **Supervised** | Ask before every action. Training mode. |
| 1 | **Guided** | Ask before destructive actions (delete, deploy, send). Read-only ops are auto-approved. |
| 2 | **Collaborative** | Only ask when confidence < 50% or task is novel. Default for new installs. |
| 3 | **Autonomous** | Only ask when confidence < 20% or facing a genuinely unknown situation. |
| 4 | **Full Autopilot** | Never ask. Handle everything. Report results only. For trusted, well-tested setups. |

- [x] **Autonomy Level Setting** - Global and per-agent autonomy levels. `pkg/agent/autonomy.go`, `pkg/config/config.go` *(Original)*
- [ ] **Autonomy Promotion** - Agent earns higher autonomy over time based on success rate. Demoted on failure. *(Original)*

## 28. Small LLM Optimization Engine (GOSqueeze)

Everything needed to make 1B-8B parameter models perform like 70B+ models. This is how iTaK Agent runs on a Dell OptiPlex with no GPU.

### Prompt Engineering for Small Models
- [x] **Prompt Compression** - Auto-compress context before sending to LLM. Summarize long tool outputs, trim irrelevant history, keep only what matters. `pkg/agent/squeeze.go` *(Original)*
- [ ] **Instruction Distillation** - Break complex instructions into atomic micro-instructions. Instead of "build a website", send "create index.html", then "add header section", etc. This is how small models succeed. *(Original)*
- [ ] **Model-Specific Prompt Templates** - Different prompt formats per model family. Qwen needs ChatML, Llama needs [INST], Mistral needs its own format. Auto-detect and apply. *(Original)*
- [ ] **Few-Shot Example Library** - Curated examples per tool/task type. Small models need 2-3 examples to understand the task. Library grows as agent learns. *(Original)*
- [ ] **System Prompt Optimization** - Auto-tune system prompts per model. Test variations, measure output quality, keep the best. DSPy-inspired. *(DSPy, Original)*
- [ ] **Negative Examples** - Show models what NOT to do. "Don't output markdown when I ask for JSON." Small models benefit heavily from explicit don'ts. *(Original)*

### Context Window Management
- [x] **Smart Sliding Window** - Prioritize recent messages + relevant older context. Not just "last N messages" but relevance-scored selection. `CompressContext()` in `squeeze.go` *(Original)*
- [x] **Tool Output Summarization** - Before feeding tool results back to LLM, summarize them. `SummarizeToolOutput()` in `squeeze.go` *(Original)*
- [x] **Context Budget Allocation** - Allocate tokens per section: 20% system prompt, 30% conversation, 30% tool results, 20% response. Per-agent `context_budget` in YAML. *(Original)*
- [ ] **Progressive Detail** - First pass: high-level overview. If model needs more detail, second pass with deeper context. Don't front-load everything. *(Original)*
- [ ] **Memory Offloading** - Move older context to vector store. Pull back only when relevant via similarity search. Infinite effective context. *(Original)*

### Structured Output Enforcement
- [ ] **Grammar-Constrained Decoding** - Force LLM output to conform to a JSON/YAML schema via GBNF grammars (llama.cpp feature). No more broken JSON. *(llama.cpp)*
- [x] **Response Validation** - Parse every LLM response. If it's supposed to be JSON and it's not, auto-retry with a simpler prompt + explicit format instructions. `ValidateJSONResponse()` in `squeeze.go` *(Original)*
- [x] **Output Repair** - Before retrying, attempt to fix common issues: strip markdown code fences, fix missing closing braces, unescape characters. `fixTrailingCommas()` in `squeeze.go` *(Original)*
- [ ] **Schema Library** - Pre-built JSON schemas for every tool call format. Models know exactly what structure to produce. *(Original)*

### Execution Optimization
- [ ] **Speculative Execution** - Run cheap/fast model first. If result passes validation, use it. If not, escalate to expensive model. Saves 80% of tokens on easy tasks. *(Original)*
- [ ] **Batch Reasoning** - Combine multiple small questions into one LLM call. "Answer these 5 questions:" instead of 5 separate calls. *(Original)*
- [ ] **Cached Responses** - Identical or near-identical prompts return cached results. Semantic cache using embeddings for fuzzy matching. *(Original)*
- [ ] **Pre-computed Decisions** - For common routing decisions (which agent handles this?), use a tiny classifier model instead of the full LLM. *(Original)*
- [ ] **Pipeline Pipelining** - While model A is generating tokens for task 1, prepare the prompt for task 2. Overlap inference with prompt construction. *(Original)*

### Model Quality Compensation
- [ ] **Verification Agents** - Pair a small "doer" model with a small "checker" model. Cheaper than one big model, often better quality. Two 4B models > one 8B model. *(Original)*
- [ ] **Consensus Voting** - For critical decisions, run the same prompt through 3 small models. Majority vote wins. *(Original)*
- [ ] **Chain of Verification (CoVe)** - After generating an answer, generate verification questions, answer them independently, then revise. Dramatically improves accuracy on small models. *(Research - Meta CoVe)*
- [ ] **Tool-Augmented Reasoning** - Instead of asking the model to calculate, give it a calculator. Instead of asking it to search, give it search. Offload what small models are bad at. *(Original)*
- [ ] **Scratchpad Reasoning** - Give models an explicit thinking/scratchpad section before the final answer. Small models produce much better outputs when they can "think out loud". *(Research - Chain of Thought)*

---

## Research Queue

Items to investigate before implementation:

### Tier 1: Adopt Now (from 2026-03-05 Research Sweep)
- [ ] **google/adk-go** - Google's official Agent Development Kit for Go. Code-first, modular multi-agent, deploys to Cloud Run. `go get google.golang.org/adk`. Study tool interface and agent composition patterns. *(https://github.com/google/adk-go)*
- [ ] **maximhq/bifrost** - Fastest Go AI gateway. 50x LiteLLM, <100us overhead, 5k RPS. MCP support, semantic caching, multi-provider fallback. Could replace custom provider routing. *(https://github.com/maximhq/bifrost)*
- [ ] **odvcencio/gotreesitter** - Pure Go tree-sitter. No CGo. Matches purego philosophy. Code analysis skill for agents. *(https://github.com/odvcencio/gotreesitter)*
- [ ] **dgraph-io/ristretto** - High-perf Go cache (SampledLFU + TinyLFU, 26.9k dependents). Prompt/embedding caching. *(https://github.com/dgraph-io/ristretto)*
- [ ] **qmuntal/stateless** - Go FSM library. Hierarchical states, external storage, DOT export. Agent state machine. *(https://github.com/qmuntal/stateless)*

### Tier 2: Patterns to Study
- [ ] **natexcvi/go-llm** - Agent framework with tools (BashTerminal, PythonREPL, KeyValueStore, JSONAutoFixer, GenericAgentTool) + memory (BufferMemory, SummarisedMemory). *(https://github.com/natexcvi/go-llm)*
- [ ] **teilomillet/gollm** - Prompt optimization, chain-of-thought, model comparison, structured JSON validation. *(https://pkg.go.dev/github.com/teilomillet/gollm)*
- [ ] **DBOS durable execution** - Extends `context.Context` for workflow checkpointing. Type-safe `RunWorkflow`/`RunAsStep`. *(https://www.dbos.dev/blog/how-we-built-golang-native-durable-execution)*
- [ ] **Datadog binary optimization** - `goda` + `go-size-analyzer` tools. 77% binary size reduction via method dead code elimination. *(https://www.datadoghq.com/blog/engineering/agent-go-binaries/)*

### Tier 3: Noted for Future
- [ ] **nalgeon/redka** - Redis re-implemented in SQL/Go. Embedded KV + persistence for agent memory. *(https://github.com/nalgeon/redka)*
- [ ] **risor.io** - Go scripting language (pipe expressions, JSON, HTTP). User-scriptable agent behaviors. *(https://risor.io/)*
- [ ] **lingrino/go-fault** - Fault injection middleware. Testing agent resilience. *(https://github.com/lingrino/go-fault)*
- [ ] **pseidemann/finish** - Zero-dep graceful shutdown for HTTP servers. *(https://github.com/pseidemann/finish)*
- [ ] **anthropics/anthropic-sdk-go** - Official Anthropic Go SDK. Claude provider. *(https://github.com/anthropics/anthropic-sdk-go)*
- [ ] **go-git/go-git** - Pure Go git implementation. Agent code management. *(https://github.com/go-git/go-git)*
- [ ] **aperturerobotics/go-quickjs-wasi-reactor** - QuickJS WASM runtime in Go. Sandboxed JS for agents. *(https://github.com/aperturerobotics/go-quickjs-wasi-reactor)*

### Original Research Queue
- [ ] **steipete/canvas** - Go-based visual workspace by OpenClaw creator. Assess relevance for iTaK Agent's Canvas Agent / dashboard whiteboard feature. *(https://github.com/steipete/canvas)*
- [ ] **steipete/wacli** - Go-based WhatsApp CLI. Evaluate for WhatsApp communication plugin. *(https://github.com/steipete/wacli)*
- [ ] **openclaw/clawhub** - ClawHub registry source code. Study for iTaK Hub marketplace architecture. *(https://github.com/openclaw/clawhub)*
- [ ] **openclaw/skills** - OpenClaw skills repository. Mine for skill pack patterns and templates. *(https://github.com/openclaw/skills)*
- [ ] **openclaw/lobster** - OpenClaw Lobster (Molty core). Study architecture patterns. *(https://github.com/openclaw/lobster)*
- [ ] **openclaw/openclaw-ansible** - Ansible deployment playbooks. Reference for iTaK Agent deployment automation. *(https://github.com/openclaw/openclaw-ansible)*
- [ ] **draw.io MCP Server** - MCP server for draw.io diagram generation. *(https://github.com/lgazo/drawio-mcp-server)*
- [ ] **Go container runtimes** - Study containerd, runc, Podman internals for Tier 2 container implementation. All written in Go. *(https://github.com/containerd/containerd, https://github.com/opencontainers/runc)*
- [ ] **gogs/gogs** - Self-hosted Git service in Go. Reference for Git hosting patterns (we'll use GitHub API instead). *(https://github.com/gogs/gogs)*
- [ ] **go-gitea/gitea** - Gogs fork, more active. Reference for Go-based Git server patterns. *(https://github.com/go-gitea/gitea)*
- [ ] **gomlx/gomlx** - Studied and forked as GOTensor. Pure Go tensor engine with XLA GPU backend. Source: `backends/simplego/`. *(https://github.com/gomlx/gomlx)*
- [ ] **tmc/langchaingo** - LangChain for Go. Fork LLM provider interface, tools, and cache layer. *(https://github.com/tmc/langchaingo)*
- [ ] **cloudwego/eino** - ByteDance LLM agent framework. Study streaming, interrupt/resume, callback patterns. *(https://github.com/cloudwego/eino)*
- [ ] **knights-analytics/hugot** - ONNX transformer pipelines in Go. Fork for ONNX model support + local embeddings. *(https://github.com/knights-analytics/hugot)*
- [ ] **kkdai/youtube** - Go YouTube downloader. Fork as base for iTaK Media YouTube extractor. *(https://github.com/kkdai/youtube)*
- [ ] **horiagug/youtube-transcript-api-go** - Go YouTube transcript extractor. Fork for iTaK Media subtitle pipeline. *(https://github.com/horiagug/youtube-transcript-api-go)*
- [ ] **gonum/gonum** - Go numeric libraries (matrices, stats, optimization). Potential dep for tensor math. *(https://github.com/gonum/gonum)*
- [ ] **vercel-labs/agent-browser** - Browser automation CLI for AI agents. Snapshot+Refs pattern, Rust CLI + Node daemon. Study architecture for iTaK Browser. *(https://github.com/vercel-labs/agent-browser)*
- [ ] **chromedp/chromedp** - Go-native Chrome DevTools Protocol client. Primary browser control layer for iTaK Browser. *(https://github.com/chromedp/chromedp)*

---

## Sources

| Project | URL | Inspiration |
|---------|-----|-------------|
| Agent Zero | https://github.com/frdel/agent-zero | Orchestrator, agents, browser, memory, nudge, projects |
| OpenClaw | https://github.com/PeterJCLaw/openclaw | Doctor, security, plugins, 50+ integrations |
| OpenClaw ClawHub | https://clawhub.ai/ | Extension marketplace, skill registry |
| OpenClaw MoltBook | https://www.moltbook.com/ | Agent social network (Reddit-style) |
| OpenFang | (dashboard ref) | Dashboard UI |
| ClickUp Super Agents | https://clickup.com/ai/agents | Agent builder, Kanban tasks, agent analytics, ambient awareness |
| VS Code Marketplace | https://marketplace.visualstudio.com/ | Extension system architecture |
| LiteLLM | https://github.com/BerriAI/litellm | Provider translation |
| BricksLLM | https://github.com/bricks-cloud/BricksLLM | Go gateway, cost tracking |
| Instawork | https://github.com/Instawork/llm-proxy | Go provider adapters |
| Plano | https://github.com/katanemo/plano | Orchestration, observability |
| Cayley | https://github.com/cayleygraph/cayley | Go knowledge graph |
| Dgraph | https://github.com/dgraph-io/dgraph | Go graph database |
| quic-go | https://github.com/quic-go/webtransport-go | WebTransport in Go |
| Docker/containerd | https://github.com/containerd/containerd | Go container runtime reference |
| runc | https://github.com/opencontainers/runc | OCI container runtime (Go) |
| Gogs | https://github.com/gogs/gogs | Self-hosted Git in Go |
| Gitea | https://github.com/go-gitea/gitea | Self-hosted Git in Go (Gogs fork) |
| Draw.io MCP | https://github.com/lgazo/drawio-mcp-server | Diagram generation via MCP |
| Canvas (Go) | https://github.com/steipete/canvas | Visual workspace in Go |
| wacli (Go) | https://github.com/steipete/wacli | WhatsApp CLI in Go |
| Vercel Browser | https://github.com/vercel-labs/agent-browser | AI browser CLI with Snapshot+Refs (iTaK Browser base) |
| chromedp | https://github.com/chromedp/chromedp | Go-native Chrome DevTools Protocol client |
| GOTensor (ex-GoMLX) | https://github.com/gomlx/gomlx | Forked as GOTensor: pure Go tensor engine, XLA GPU backend, training/finetuning |
| LangChainGo | https://github.com/tmc/langchaingo | LLM providers, tools, cache |
| Eino | https://github.com/cloudwego/eino | ByteDance agent framework, streaming, interrupt/resume |
| Hugot | https://github.com/knights-analytics/hugot | ONNX transformer pipelines in Go |
| Gonum | https://github.com/gonum/gonum | Go numeric/matrix libraries |
| kkdai/youtube | https://github.com/kkdai/youtube | Go YouTube downloader (iTaK Media base) |
| yt-transcript-api-go | https://github.com/horiagug/youtube-transcript-api-go | Go YouTube transcript extraction |
| Ory Hydra | https://github.com/ory/hydra | OAuth2/OIDC server (iTaK Auth base) |
| Ory Kratos | https://github.com/ory/kratos | Identity management (iTaK Auth base) |
| Fyne | https://github.com/fyne-io/fyne | Go cross-platform GUI (iTaK Auth mobile app) |
| Bubbletea | https://github.com/charmbracelet/bubbletea | Go TUI framework (CLI wizard) |
| Google A2A | https://github.com/google/A2A | Agent-to-Agent protocol |
| Google ADK-Go | https://github.com/google/adk-go | Official Go Agent Development Kit |
| Bifrost | https://github.com/maximhq/bifrost | Fastest Go AI gateway (<100us overhead) |
| gotreesitter | https://github.com/odvcencio/gotreesitter | Pure Go tree-sitter runtime (no CGo) |
| Ristretto | https://github.com/dgraph-io/ristretto | High-perf Go cache (SampledLFU + TinyLFU) |
| Stateless | https://github.com/qmuntal/stateless | Go finite state machine library |
| go-llm | https://github.com/natexcvi/go-llm | Go agent framework with tools + memory |
| gollm | https://github.com/teilomillet/gollm | Go LLM framework with prompt optimization |
| Redka | https://github.com/nalgeon/redka | Redis re-implemented in SQL (Go) |
| Risor | https://risor.io/ | Go scripting language |
| go-fault | https://github.com/lingrino/go-fault | Go fault injection library |
| finish | https://github.com/pseidemann/finish | Zero-dep graceful shutdown |
| Anthropic Go SDK | https://github.com/anthropics/anthropic-sdk-go | Official Anthropic SDK for Go |
| HackBrowserData | https://github.com/moonD4rk/HackBrowserData | Go browser data extraction (cookies, passwords, history) |

---

## iTaK Shield

### PRIVATE: iTaK Shield (AI-Powered Physical Security Monitoring)

> **Commercial product. Private repo.** Built on the public iTaK ecosystem.

AI agent swarm that replaces call center operators for physical security monitoring. One iTaK Agent instance monitors 1000+ cameras, network devices, and solar-powered surveillance units.

**Repo:** [iTaK Shield](https://github.com/David2024patton/iTaK Shield) (private)

### Core Capabilities
- [ ] **Device Fleet Monitoring** - Poll 1000+ IPs via SNMP/HTTP/ICMP/Zabbix. Axis cameras, Cradlepoint routers, MicroHard LTE, Morningstar solar controllers, PoE switches, NVRs
- [ ] **Vision Event Detection** - iTaK Torch runs moondream2/Qwen3-VL on camera snapshots. Detect people, vehicles, animals. Annotate screenshots with bounding boxes, labels, confidence scores, timestamps
- [ ] **Multi-Camera Tracking** - Cross-reference detections across feeds. PTZ auto-slew to events. False positive learning per site
- [ ] **Solar Power Optimization** - Correlate weather/GPS/season with Morningstar harvest data. Auto-generate optimized charge profiles. Predict battery depletion windows
- [ ] **Automated Reporting** - Daily health, weekly trends, monthly optimization recommendations. Natural language via iTaK Gateway LLMs
- [ ] **SaaS Multi-Tenant Portal** - Client dashboard with live health, event feed with annotated images, report downloads, alert configuration
- [ ] **Agent Swarm Scaling** - Scout swarm (1 per ~100 devices), Vision workers (GPU-bound), Researcher (weather/solar), Doctor (fault diagnosis), Reporter (NLG), Alert Manager (escalation chains)

### Revenue Model
- Per-device/month, per-site/month, or tiered (Basic monitoring / Pro vision+events / Enterprise optimization+reports)

### Target Hardware
ECAM MSU stack: Axis cameras (fixed + PTZ), Cradlepoint IBR, MicroHard pMDDL, Morningstar controllers, PoE speakers, managed switches, environmental sensors

### Zabbix Integration
Dual-mode monitoring: use Zabbix API as a data backend where clients already run it (no duplicate polling), or iTaK Shield's native Go polling engine for lightweight single-binary deploys. Community templates provide pre-built SNMP OID mappings for hundreds of device types.

**References:**
| Repo | Use |
|---|---|
| [zabbix/zabbix](https://github.com/zabbix/zabbix) | Core - SNMP OID mappings, trigger logic, device state detection patterns |
| [zabbix-docker](https://github.com/zabbix/zabbix-docker) | Deployment - Docker compose for bundling Zabbix alongside iTaK Shield |
| [community-templates](https://github.com/zabbix/community-templates) | Gold mine - Pre-built templates for Cradlepoint, Axis, network gear OIDs |
| [jjmartres/Zabbix](https://github.com/jjmartres/Zabbix) | Custom templates and monitoring scripts |
| [itmicus/zabbix](https://github.com/itmicus/zabbix) | Additional community device templates |

### Network Security Monitoring
Protect client networks from hackers and vulnerabilities using a dual-layer approach:
- [ ] **DNS Sinkhole** - Route client DNS through iTaK Shield. Log queries, block known malicious/C2 domains, detect unusual DNS patterns (DGA, tunneling). Lightweight, no agents required
- [ ] **Endpoint Agent** - Lightweight Go agent on client servers/endpoints. File integrity monitoring, vulnerability scanning, log collection, anomaly detection (Wazuh/OSSEC style)
- [ ] **Network Flow Analysis** - Ingest NetFlow/sFlow from managed switches. Detect port scans, lateral movement, data exfiltration, unusual traffic patterns
- [ ] **Vulnerability Scanning** - Scheduled scans of client IP ranges. CVE matching against known services. Auto-generate remediation reports
- [ ] **Threat Intel Feeds** - Integrate with open threat intelligence (AlienVault OTX, Abuse.ch, VirusTotal API). Correlate against client traffic

### Intrusion Detection & Attack Tracing
Automated detection, tracing, and documentation of network attacks:
- [ ] **Auto-Capture Attacker IPs** - Log all malicious connection attempts (SYN floods, port scans, brute force, credential stuffing). Real-time source IP capture from endpoint firewall/IDS logs
- [ ] **Reverse DNS + WHOIS Enrichment** - Auto-lookup every attacking IP: owner ISP/hosting, ASN, geolocation (city/country), abuse contact email. Cross-reference with threat intel databases
- [ ] **VPN/Proxy/Tor Detection** - Check attacker IPs against known VPN provider ranges, Tor exit node lists, datacenter IP databases. Flag as "likely masked" with confidence level
- [ ] **Behavioral Fingerprinting (JA3/JA4)** - Even behind VPNs, attackers leave fingerprints: TLS client hello (JA3/JA4 hashes), tool signatures (nmap, hydra, metasploit), timing patterns, HTTP headers. Correlate attacks across different source IPs to same actor
- [ ] **Honeypot Deployment** - iTaK Shield agents auto-spin fake services (SSH, RDP, ONVIF cameras, web portals) on unused IPs. Attackers interact, revealing tools and techniques. Occasionally capture real IPs when attackers slip up
- [ ] **Automated Incident Reports** - Generate forensic-grade incident reports with full evidence (timestamps, IPs, enrichment data, attack timeline). Ready for law enforcement referral or cyber insurance claims
- [ ] **Automated Abuse Reports** - Auto-send abuse complaints to attacker's ISP/hosting provider with evidence packet

### Active Defense System (Automated Response)
AI-driven agents that don't just detect attacks - they stop them in real-time:
- [ ] **Auto-Block via Firewall API** - On detecting attack patterns (brute force, port scan, SYN flood), agent pushes DROP rules directly to client firewalls (iptables, pfSense, Cradlepoint API). Attacker blocked in seconds, no human needed
- [ ] **Adaptive Pattern Recognition** - AI detects slow/distributed attacks that evade simple rate limits (1 port/minute scans, credential stuffing from 50+ rotating IPs). Correlates via JA3 fingerprint and blocks all related IPs at once
- [ ] **Honeypot Tripwire** - Any IP that touches a honeypot gets instantly blacklisted across the entire client fleet
- [ ] **Outbound Exfil Detection** - Monitor for unusual outbound traffic patterns (large uploads, connections to known C2 IPs, DNS tunneling). Auto-throttle or block suspicious outbound connections
- [ ] **CVE Auto-Response** - When new CVEs drop for deployed hardware (Cradlepoint, Axis, etc.), auto-scan all client devices and push priority alerts with remediation steps

### Collective Defense Network (Fleet Advantage)
The killer SaaS differentiator - every client makes every other client safer:
- [ ] **Cross-Client Threat Sharing** - Attacker blocked at Client A -> proactively blocked at Clients B, C, D within seconds
- [ ] **Private Threat Intel Feed** - iTaK Shield builds its own threat database from real attack data across all clients
- [ ] **Network Effect Scaling** - More clients = better protection. Each new client adds intelligence to the collective defense
- [ ] **Industry-Specific Patterns** - Recognize attack campaigns targeting security/surveillance industry specifically

### ECAM Hardware Defense Matrix
Device-specific active defense for MSU deployments:

| Device | Defense Actions |
|---|---|
| **Cradlepoint** | Monitor via API, auto-block IPs on built-in firewall, detect SIM swap, alert on unusual cellular tower connections |
| **Smart Switch** | Monitor port states, detect rogue devices on unused ports, auto-disable compromised ports |
| **Axis Cameras** | Monitor ONVIF/RTSP auth, block brute force, detect firmware tampering, alert on config changes |
| **PoE Speaker** | Monitor for unauthorized access, prevent audio hijacking |
| **Morningstar** | Alert on unexpected Modbus commands, detect charge profile tampering |

### Client Dashboard
What the client sees on their SaaS portal:
- [ ] **Live Attack Map** - Real-time visualization of blocked attacks with source geolocations
- [ ] **Threat Score** - Current threat level with active attack count ("3 active attacks. All blocked.")
- [ ] **Monthly Security Report** - "iTaK Shield blocked 14,782 malicious connections from 847 unique IPs across 23 countries. 0 successful breaches." - This report is what keeps clients paying
- [ ] **Incident Timeline** - Chronological view of all security events with drill-down to forensic details

### Licensing Model Options
- **SaaS** - Per-device/month or per-site/month recurring revenue. Client accesses dashboard, you manage the backend
- **Proprietary License** - One-time perpetual license + annual maintenance/support fee. Client runs iTaK Shield on their own infrastructure
- **Managed Service** - You operate iTaK Shield on behalf of the client (SOC-as-a-Service). Highest margin, most involvement

### Novel "First and Only" Features
Capabilities no existing security platform offers:
- [ ] **Physical-Cyber Event Correlation** - Camera offline at 2:47 AM + network brute force at 2:48 AM = coordinated attack. Correlate physical events (feed loss, motion, door sensors) with cyber events (port scans, login attempts, firewall probes). Nobody else does this
- [ ] **Firmware Supply Chain Verification** - Hash-check camera/device firmware updates against known-good signatures before deployment. Block unauthorized/compromised firmware (SolarWinds-style supply chain attacks on physical security devices)
- [ ] **Cellular/LTE Security** - Monitor Cradlepoint cellular connections for: IMSI catcher (Stingray) detection, SIM cloning/swapping, rogue base stations, cellular MITM attacks. Completely unmonitored attack surface for every remote deployment
- [ ] **Camera Insider Threat Detection** - Detect suspicious config changes by authorized users: viewing angle changes, disabled motion detection, altered recording schedules, muted audio. Actions taken BEFORE committing a crime
- [ ] **Autonomous Self-Healing** - For unmanned sites: auto-isolate compromised devices (disable switch port), restart services, rollback to last-known-good config, maintain other operations, alert SOC. All without human intervention
- [ ] **Power Infrastructure Security** - Secure Morningstar Modbus (zero auth industrial protocol) and EFOY fuel cell management interfaces. Attacker kills power = entire site goes dark without touching a camera
- [ ] **Compliance Report Auto-Generation** - Auto-generate NIST 800-53, SOC 2, PCI DSS compliance reports from monitoring data. Saves clients $50K+ in consulting fees. Revenue stream on its own
- [ ] **Digital Chain of Custody** - Cryptographically sign video exports with timestamps, hash verification, tamper detection. Forensic-grade evidence admissible in court and valid for insurance claims

### AI Video Analytics (iTaK Torch-Powered)
On-device AI inference using iTaK Torch for real-time video analysis across camera feeds:

#### Cross-Camera Re-Identification (Re-ID)
Track the same target across multiple camera views. Core system that powers all tracking analytics:
- [ ] **Person Re-ID** - Track individuals across cameras using appearance features (clothing, body shape, gait). No facial recognition required. Follow a person from Camera 1 parking lot -> Camera 3 entrance -> Camera 7 hallway
- [ ] **Vehicle Re-ID** - Track vehicles by make, model, color, and visual features across cameras. Correlate with ALPR for positive ID
- [ ] **Animal/Wildlife Re-ID** - Species classification and individual tracking. Critical for conservation sites, farms, and construction sites near wildlife areas
- [ ] **Re-ID Timeline** - Generate complete movement timeline for any tracked target: "Subject first seen Camera 2 at 14:32, moved to Camera 5 at 14:35, exited via Camera 9 at 14:41"
- [ ] **Cross-Site Re-ID** - Track targets across different client sites in the collective defense network. Shoplifter hits Store A, flagged at Store B automatically

#### License Plate Recognition (ALPR/ANPR)
- [ ] **Automatic Plate Reading** - Read plates from camera feeds in real-time. Support for US, EU, and international formats
- [ ] **Watchlist Matching** - Alert on known plates (stolen vehicles, banned persons, VIP arrivals)
- [ ] **Parking/Access Control** - Auto-authorize vehicles by plate for gated entries
- [ ] **Historical Plate Search** - "Show me every time plate ABC-1234 appeared in the last 30 days"

#### Behavioral Analytics
- [ ] **Loitering Detection** - Alert when person/vehicle remains in an area beyond threshold time
- [ ] **Perimeter Breach** - Detect unauthorized zone entry with directional tracking (crossed fence from outside to inside)
- [ ] **Unusual Movement** - Running, crawling, erratic paths, backtracking. Patterns that deviate from normal foot traffic
- [ ] **Tailgating Detection** - Multiple people entering through a single-authorization door access
- [ ] **Wrong Direction** - Person/vehicle moving against expected flow (entering through an exit)

#### Threat Detection
- [ ] **Weapon Detection** - Real-time detection of firearms, knives, and other weapons in camera feeds
- [ ] **Smoke/Fire Detection** - Early visual detection before traditional sensors trigger. Critical for unmanned sites
- [ ] **Abandoned Object** - Detect bags, packages, or objects left unattended (bomb threat protocol)
- [ ] **PPE Compliance** - Hard hats, high-vis vests, safety glasses detection for construction/industrial sites
- [ ] **Crowd Density** - Monitor crowd size, detect overcrowding, alert when thresholds exceeded

#### Intelligence & Analytics
- [ ] **Heatmaps** - Traffic flow visualization across cameras. Where do people/vehicles go most? Useful for retail, security, and site planning
- [ ] **Dwell Time Analytics** - How long do people spend in specific zones? Retail conversion, security risk scoring
- [ ] **Occupancy Counting** - Real-time count of people in zones. Fire code compliance, capacity management
- [ ] **Time-of-Day Patterns** - "Parking lot is busiest 7-9 AM. No activity after 11 PM. Activity at 2 AM = anomaly"
- [ ] **Incident Video Compilation** - Auto-compile video clips from all cameras that captured a tracked target into a single incident timeline

### Audio Analytics (Paired with Camera Feeds)
- [ ] **Gunshot Detection** - Acoustic classification of gunfire. Auto-trigger camera Re-ID to find source visually. Alert with GPS coordinates
- [ ] **Glass Break Detection** - Breaking glass sound classification. Trigger nearest camera PTZ to focus on source
- [ ] **Distress/Scream Detection** - Detect screaming, yelling for help, aggressive voice tones
- [ ] **Vehicle Collision Detection** - Crash sounds trigger auto-recording, incident report generation, emergency dispatch alert
- [ ] **Audio-Visual Correlation** - Match audio events to camera-detected events for higher confidence alerting

### Drone Detection & Tracking
- [ ] **Visual Drone Detection** - AI identification of drones in camera feeds using iTaK Torch inference
- [ ] **RF Signal Detection** - Monitor for drone controller frequencies (2.4GHz, 5.8GHz, DJI protocols)
- [ ] **Flight Path Tracking** - Track drone movement across cameras, predict landing/return path
- [ ] **Airspace Violation Alerts** - Define protected airspace zones, alert on any drone incursion
- [ ] **Counter-Drone Logging** - Full forensic log of drone events for law enforcement (time, path, RF signature)

### Dark Web Monitoring
- [ ] **Credential Leak Scanning** - Monitor dark web forums, paste sites, and marketplaces for client credentials (emails, passwords, VPN creds)
- [ ] **Network Access Sales** - Detect if access to client's network is being sold (RDP access, VPN accounts)
- [ ] **Data Breach Detection** - Alert when client's proprietary data surfaces on dark web
- [ ] **Threat Actor Tracking** - Monitor threat actors who have previously targeted client infrastructure

### Access Control Integration
- [ ] **Badge/Card Reader Integration** - Tie into HID, Lenel, and other access control systems via API
- [ ] **Badge + Camera Correlation** - Badge swipe at Door A + person on Camera B = verified identity. Badge swipe but no person visible = cloned badge alert
- [ ] **Tail Detection** - One badge swipe, two people entered. Detected via camera + access log mismatch
- [ ] **Unauthorized Area Detection** - Visitor or employee enters zone their badge doesn't authorize. Camera Re-ID + badge zone comparison

### Environmental Monitoring
- [ ] **Temperature Alerting** - Monitor server rooms, equipment closets, outdoor enclosures. Alert on threshold breach
- [ ] **Water/Flood Detection** - Sensor integration for water leak detection in critical areas
- [ ] **Power Monitoring** - UPS status, battery levels, mains power loss. Auto-alert on power events
- [ ] **HVAC Correlation** - Temp spike + HVAC offline = equipment at risk. Auto-notify maintenance

### Backup & Recording Verification
- [ ] **NVR Health Monitoring** - Verify NVR is recording, disk not full, no silent failures
- [ ] **Recording Gap Detection** - "Camera 7 hasn't recorded in 48 hours despite being online" - prevent the "we needed footage but the drive was full" disaster
- [ ] **Automatic Backup Verification** - Verify backup integrity on schedule. Hash-check recorded footage
- [ ] **Storage Forecasting** - Predict when storage will fill based on current usage rates. Auto-alert before it happens

### Government-Grade Security (Code-Only, Zero Cost)
All features implementable in code. No paid certifications required to build - certifications only needed when actually selling to government.

#### Cryptography & Data Protection
- [ ] **FIPS 140-2/3 Compliant Crypto** - Use Go's BoringCrypto module (Google's FIPS-validated crypto). AES-256-GCM encryption, TLS 1.3 only, ECDSA/RSA-4096 signing. All free in Go stdlib
- [ ] **Encryption Everywhere** - All data encrypted at rest (AES-256) and in transit (TLS 1.3). No plaintext storage ever. Database encryption, config encryption, log encryption
- [ ] **Key Management** - Built-in key rotation, key escrow, and key destruction. HSM-ready architecture (plug in hardware when client has it)
- [ ] **Data Classification Labels** - Tag all data as Unclassified, CUI, Sensitive, Restricted. Enforce handling rules per classification level
- [ ] **Secure Deletion** - DoD 5220.22-M compliant data wiping. Multi-pass overwrite when data is deleted

#### Authentication & Access Control
- [ ] **CAC/PIV Smart Card Auth** - Support Common Access Card (military) and Personal Identity Verification (civilian) via PKCS#11 interface. No username/password
- [ ] **Multi-Factor Authentication** - TOTP, FIDO2/WebAuthn, smart card. Enforce MFA for all access
- [ ] **Role-Based Access Control (RBAC)** - Granular permissions: Admin, Operator, Viewer, Auditor, Maintenance. Each role sees only what they need
- [ ] **Clearance-Level Filtering** - Users only see data matching their clearance level. Secret-cleared user can't see Top Secret data
- [ ] **Session Management** - Automatic timeout (15 min gov standard), concurrent session limits, forced re-auth for sensitive actions
- [ ] **Password Policy Engine** - Minimum 15 chars, complexity requirements, password history (24 previous), lockout after 3 failed attempts. All configurable per STIG

#### Audit & Compliance
- [ ] **STIG-Compliant Audit Trail** - Every action logged: who, what, when, from where, success/fail. Tamper-proof append-only logs with cryptographic chaining (blockchain-style integrity)
- [ ] **7-Year Log Retention** - Compressed, encrypted, indexed log storage. Government requires minimum 7 years
- [ ] **Audit Log Export** - Export in Syslog, CEF, LEEF, JSON formats for SIEM integration (Splunk, Elastic, ArcSight)
- [ ] **Change Detection** - Log every configuration change with before/after state and who made it
- [ ] **Compliance Dashboard** - Auto-map current security posture against NIST 800-53, NIST 800-171 (CUI), CJIS Security Policy. Show pass/fail per control
- [ ] **NDAA Section 889 Scanner** - Scan network for prohibited manufacturers (Huawei, ZTE, Hikvision, Dahua, Hytera). Auto-flag non-compliant devices. Generate remediation report

#### Air-Gap & Offline Operation
- [ ] **100% Offline Mode** - Full functionality with zero internet. No cloud calls, no telemetry, no update checks. All AI via iTaK Torch local inference
- [ ] **Offline Threat Intel Updates** - Load threat intelligence feeds via physical media (USB). Verify media integrity before import
- [ ] **Local-Only AI** - All ML models run on-device. No data leaves the network. Ever
- [ ] **Sneakernet Update System** - Software updates delivered via signed, encrypted USB packages. Verify digital signature before installing

#### Government Facility Features
- [ ] **Visitor Management System** - Check-in, badge issuance, escort assignment, movement tracking via Re-ID, check-out verification. Alert if visitor leaves escort
- [ ] **Duress/Panic System** - Silent alarm activation (hardware panic button or software trigger). Auto-lockdown, camera focus on distress location, silent SOC notification
- [ ] **SCIF Zone Protection** - Detect unauthorized wireless devices (phones, cameras, recording equipment) brought into restricted areas via RF monitoring
- [ ] **Screen Watermarking** - Display classification level, user identity, and timestamp on all screens. Makes unauthorized photography traceable
- [ ] **Continuity of Operations (COOP)** - Automatic failover to backup system. Geographic redundancy support. Recovery Time Objective < 60 seconds
- [ ] **Secure Multi-Tenancy** - Multiple agencies/departments share infrastructure with complete data isolation. No cross-tenant data leakage possible
- [ ] **Evidence Locker** - Secured, tamper-proof storage for forensic evidence. Chain of custody tracking, access logging, integrity verification. Court-admissible

#### Future-Proofing
- [ ] **Post-Quantum Cryptography** - Implement NIST PQ standards (ML-KEM/Kyber, ML-DSA/Dilithium) alongside current crypto. Ready for quantum computing threats before they arrive
- [ ] **Plugin Architecture** - Modular system allows adding new device types, protocols, analytics models without core changes
- [ ] **API-First Design** - Every feature accessible via authenticated REST API. Future UIs, integrations, and third-party tools plug in cleanly
- [ ] **Model Hot-Swap** - Replace AI models (Re-ID, weapon detection, ALPR) without restarting the system. Drop in newer/better models as they become available
- [ ] **Protocol Extensibility** - ONVIF, RTSP, Modbus, SNMP, BACnet, MQTT today. New protocols added as plugins. Industry standards change, iTaK Shield adapts
- [ ] **Multi-Architecture** - Compiled for x86_64, ARM64 (Raspberry Pi, Jetson, Apple Silicon), RISC-V. Go cross-compilation makes this free

### Blockchain Integrity Chain
Use cryptographic chaining to make iTaK Shield's own data tamper-proof:
- [ ] **Immutable Audit Chain** - Every log entry's hash includes the previous entry's hash. Tamper with one log = chain breaks, immediately detectable. Blockchain-style integrity without a blockchain
- [ ] **Evidence Timestamping** - Anchor forensic evidence hashes to public blockchains (Bitcoin OP_RETURN, Ethereum) as cryptographic proof that evidence existed at a specific time. Court-admissible, unforgeable
- [ ] **Configuration Integrity Chain** - Every device config change chained cryptographically. Prove exact state of every device at any point in history
- [ ] **Firmware Signature Chain** - Track entire firmware version history per device. Prove no unauthorized firmware was ever installed
- [ ] **Cross-Node Consensus** - Multiple iTaK Shield nodes verify each other's chains. Single compromised node can't rewrite history without other nodes detecting it

### Crypto Threat Monitoring
Detect cryptocurrency-related threats on client networks:
- [ ] **Crypto Mining Detection** - Detect unauthorized mining: CPU/GPU usage anomalies, connections to known mining pools (Stratum protocol), ethash/randomx traffic patterns. Employees or hackers hijacking client hardware
- [ ] **Cryptojacking Detection** - Browser-based mining scripts (Coinhive successors) running on client machines. Detect via network traffic to WebSocket mining proxies
- [ ] **Ransomware Wallet Tracking** - On ransomware incident, trace Bitcoin/Monero wallet addresses through blockchain analysis. Map payment flows, identify exchange cashout points
- [ ] **Sanctioned Wallet Detection** - Monitor for network traffic to wallets on OFAC sanctions list. Compliance requirement for government and financial clients
- [ ] **DeFi/Smart Contract Monitoring** - Detect unauthorized smart contract interactions from client network. Flag connections to known rug-pull contracts, mixer services (Tornado Cash), or laundering protocols

### Blockchain Network Protection
Protect companies that operate blockchain infrastructure (exchanges, DeFi protocols, validators, Web3 platforms):

#### Node & Infrastructure Security
- [ ] **Node Health Monitoring** - Monitor blockchain nodes (Ethereum, Bitcoin, Solana, Polygon, etc.) for uptime, sync status, peer count, block height lag. Alert on desync or stalled nodes
- [ ] **RPC Endpoint Protection** - Secure JSON-RPC, WebSocket, and gRPC endpoints. Rate limiting, authentication enforcement, DDoS protection. Block unauthorized access to node APIs
- [ ] **Validator Defense** - Monitor validator nodes for slashing conditions, missed attestations, double-signing attempts. Alert before penalties hit. Protect validator keys
- [ ] **P2P Network Monitoring** - Track peer connections for eclipse attacks (attacker surrounds your node with malicious peers to isolate it from the network). Detect suspicious peer behavior
- [ ] **Mempool Surveillance** - Monitor pending transaction pool for sandwich attacks, front-running bots, and MEV extraction targeting client's transactions

#### Wallet & Key Security
- [ ] **Hot Wallet Monitoring** - Real-time alerts on unexpected withdrawals, unusual transaction sizes, transfers to unknown addresses. Configurable thresholds per wallet
- [ ] **Private Key Exposure Detection** - Scan code repos, logs, configs, and environment variables for accidentally exposed private keys. Instant alert + auto-revoke if possible
- [ ] **Multi-Sig Enforcement** - Verify multi-signature requirements are met before transactions execute. Alert on attempts to bypass multi-sig
- [ ] **Whale Movement Alerts** - Track large value movements in/out of client wallets. Detect potential theft in progress
- [ ] **Address Poisoning Detection** - Detect when attackers create lookalike addresses to trick users into sending funds to wrong wallet

#### Smart Contract Defense
- [ ] **Contract Interaction Monitoring** - Track all interactions with client's deployed smart contracts. Alert on unusual patterns, new contract callers, or unexpected function calls
- [ ] **Reentrancy Detection** - Monitor for reentrancy attack patterns against client contracts in real-time
- [ ] **Flash Loan Attack Detection** - Detect flash loan sequences targeting client's DeFi protocols (large borrow -> price manipulation -> profit -> repay in one block)
- [ ] **Governance Attack Monitoring** - Track voting power accumulation, proposal submissions, and execution. Alert on hostile governance takeover attempts
- [ ] **Bridge Security** - Monitor cross-chain bridge operations for unauthorized mints, double-spends, or oracle manipulation

#### Compliance & Forensics
- [ ] **Transaction Graph Analysis** - Map the flow of funds through client's ecosystem. Identify suspicious patterns, circular transactions, wash trading
- [ ] **AML/KYC Enforcement Integration** - Flag transactions involving addresses linked to sanctioned entities, darknet markets, or known fraud rings (using Chainalysis/Elliptic-style analysis)
- [ ] **Regulatory Reporting** - Auto-generate SAR (Suspicious Activity Reports) and CTR (Currency Transaction Reports) from detected anomalies
- [ ] **Incident Forensics** - Full chain-of-events reconstruction after a hack: what was exploited, how funds moved, where they went, can they be recovered

### Financial Institution Network Protection
Protect banks, credit unions, and financial service providers. One of the highest-paying security verticals.

#### Transaction & Fraud Monitoring
- [ ] **Real-Time Transaction Anomaly Detection** - AI monitors transaction patterns per account. Flag anomalies: unusual amounts, frequency, locations, recipient patterns. "Account averaging $200 transfers just sent $47,000 to an offshore account"
- [ ] **Wire Transfer / SWIFT Monitoring** - Monitor SWIFT messages and wire transfers for unauthorized or manipulated transactions. Alert on transactions to sanctioned countries or known fraud accounts
- [ ] **Card Skimmer Detection** - Monitor ATM and POS terminal network connections for unauthorized data exfiltration (card skimmers phone home to attacker servers)
- [ ] **Account Takeover Detection** - Detect credential stuffing, SIM swap attacks, and social engineering attempts against customer accounts
- [ ] **Insider Trading Correlation** - Monitor for unusual data access patterns before major financial events. Employee accessing accounts they don't normally touch

#### Branch & ATM Physical Security
- [ ] **ATM Tamper Detection** - Camera-based detection of skimmer installation, card trap devices, shoulder surfing. Correlate with ATM network events
- [ ] **Branch Camera Analytics** - Loitering outside vault, after-hours presence, safe zone violations. Cross-camera Re-ID for tracking suspects across branches
- [ ] **Night Drop Monitoring** - Camera monitoring of after-hours deposit drops. Detect tampering, theft attempts, or suspicious behavior
- [ ] **Drive-Through Security** - License plate + facial tracking at drive-through lanes. Correlate with transaction data for fraud investigation
- [ ] **Vault Access Monitoring** - Dual-control verification via camera: two authorized personnel required, camera confirms both present. Alert if vault opened by single person

#### Regulatory Compliance (Banking-Specific)
- [ ] **PCI DSS Compliance** - Auto-validate Payment Card Industry Data Security Standard controls. Cardholder data environment segmentation verification, access logging, encryption status
- [ ] **GLBA Safeguards Rule** - Gramm-Leach-Bliley Act compliance monitoring. Track access to customer financial data, verify encryption, audit trail for all customer record access
- [ ] **SOX Monitoring** - Sarbanes-Oxley compliance. Monitor financial system access controls, segregation of duties, change management on financial applications
- [ ] **FFIEC CAT Assessment** - Auto-generate Federal Financial Institutions Examination Council Cybersecurity Assessment Tool responses from monitoring data
- [ ] **BSA/AML Automated Filing** - Bank Secrecy Act compliance. Auto-generate Suspicious Activity Reports (SARs) and Currency Transaction Reports (CTRs) when thresholds are met

#### Financial Network Segmentation
- [ ] **SWIFT Network Isolation** - Monitor and enforce isolation of SWIFT terminals from general network. Alert on any unauthorized connection attempts to/from SWIFT zone
- [ ] **Cardholder Data Environment (CDE) Monitoring** - Continuous verification that CDE network segments remain isolated. Detect any cross-segment traffic violations
- [ ] **Core Banking System Protection** - Monitor connections to core banking platforms (FIS, Fiserv, Jack Henry). Detect unauthorized access, unusual query patterns, data extraction attempts
- [ ] **Third-Party Vendor Access Control** - Monitor vendor remote access sessions in real-time. Record, audit, and terminate if suspicious. Many bank breaches come through vendor connections

### Monitoring & Dashboard Resources
Reference implementations for building iTaK Shield's Grafana-based dashboards:

| Repository | Purpose |
|---|---|
| [grafana/grafana-zabbix](https://github.com/grafana/grafana-zabbix) | Official Zabbix plugin for Grafana. Reference for building custom iTaK Shield Grafana data source plugins. Shows plugin architecture, data source config, query builders, and alerting integration |

#### Grafana Plugin Development Notes
- iTaK Shield can expose its data as a Grafana data source plugin (like grafana-zabbix does)
- This means clients with existing Grafana deployments can add iTaK Shield as a data source alongside their other monitoring
- Plugin architecture: React frontend + Go backend (perfect - we already use Go)
- Enables custom dashboards: attack maps, camera status grids, threat scores, compliance scorecards
- Can also build standalone iTaK Shield dashboards using Grafana's embedding/iframe mode

### Healthcare Network Protection (HIPAA)
Protect hospitals, clinics, health systems, and medical device networks.

#### Patient Data (PHI) Protection
- [ ] **PHI Access Monitoring** - Track every access to Protected Health Information. Who viewed which patient records, when, from where. Alert on unusual access patterns (nurse accessing celebrity patient records, bulk record exports)
- [ ] **EHR System Protection** - Monitor Electronic Health Record systems (Epic, Cerner, Meditech) for unauthorized access, data exports, and privilege escalation
- [ ] **Medical Device Network Isolation** - Monitor and enforce segmentation between medical devices (infusion pumps, ventilators, MRI machines) and general IT network. FDA requires this
- [ ] **DICOM/PACS Security** - Monitor medical imaging traffic (X-rays, MRIs, CTs). Detect unauthorized access to imaging servers, data exfiltration of patient scans
- [ ] **Prescription Monitoring** - Detect unusual prescription patterns that may indicate fraud, diversion, or compromised credentials (doctor account writing 100 opioid prescriptions at 3 AM)

#### Medical Device Security
- [ ] **IoMT Device Inventory** - Auto-discover and inventory all Internet of Medical Things devices on the network. Track firmware versions, known vulnerabilities, communication patterns
- [ ] **Infusion Pump Monitoring** - Detect unauthorized dosage changes, firmware tampering, or network attacks on connected infusion pumps. Patient safety critical
- [ ] **Implant Communication Security** - Monitor wireless communications to/from connected implants (pacemakers, insulin pumps). Detect replay attacks or unauthorized commands
- [ ] **Biomedical Equipment Alerts** - Monitor critical life-support equipment uptime. Ventilator offline, heart monitor disconnected = immediate escalation with camera verification of the room

#### Hospital Physical Security
- [ ] **Infant Abduction Prevention** - Camera Re-ID tracking of infants (via ankle tag correlation) and adults in maternity ward. Alert on infant moving toward exits without authorized personnel
- [ ] **Pharmacy Access Control** - Camera + badge verification for controlled substance access. Dual-control enforcement via video. Detect tailgating into pharmacy
- [ ] **Emergency Department Monitoring** - Weapon detection, aggressive behavior recognition, loitering. Protect staff and patients in high-risk areas
- [ ] **Patient Elopement Detection** - Track at-risk patients (dementia, psych holds) via camera Re-ID. Alert if patient approaches exits or restricted areas
- [ ] **Surgical Suite Integrity** - Camera verification of authorized personnel only in OR. PPE compliance (gowns, masks, gloves). Detect unauthorized entry

#### HIPAA Compliance Automation
- [ ] **HIPAA Security Rule Mapping** - Auto-map iTaK Shield controls to HIPAA Security Rule requirements (Administrative, Physical, Technical safeguards). Show pass/fail per requirement
- [ ] **Breach Notification Generator** - On detected PHI breach, auto-generate HHS breach notification with all required fields (individuals affected, data types, timeline). 60-day notification deadline countdown
- [ ] **Business Associate Monitoring** - Track third-party vendor access to PHI systems. Verify BAA compliance, audit vendor sessions, alert on scope violations
- [ ] **Risk Assessment Automation** - Auto-generate HIPAA-required annual risk assessment from monitoring data. Identify gaps, recommend remediation

### Energy & Utilities Network Protection (NERC CIP)
Protect power plants, substations, water treatment, oil/gas, and critical infrastructure.

#### SCADA/ICS Protection
- [ ] **SCADA Protocol Monitoring** - Deep packet inspection for Modbus, DNP3, IEC 61850, IEC 104, OPC-UA. Detect unauthorized commands, parameter changes, or reconnaissance
- [ ] **PLC/RTU Protection** - Monitor Programmable Logic Controllers and Remote Terminal Units for unauthorized firmware uploads, logic changes, or reboot commands
- [ ] **HMI Access Control** - Track who accesses Human-Machine Interfaces. Alert on unauthorized logins, unusual command sequences, or access from unexpected locations
- [ ] **Process Value Monitoring** - Track critical process values (voltage, pressure, flow rate, temperature). AI learns normal ranges and alerts on anomalies that could indicate attack or equipment failure
- [ ] **Safety Instrumented System (SIS) Protection** - Monitor safety systems separately from process control. Detect attempts to disable safety mechanisms (Triton/TRISIS attack pattern)

#### Smart Grid & Power Monitoring
- [ ] **AMI/Smart Meter Security** - Monitor Advanced Metering Infrastructure for meter tampering, unauthorized firmware, command injection, and data manipulation
- [ ] **DER Security** - Protect Distributed Energy Resources (solar inverters, battery storage, wind turbines). Monitor for unauthorized control commands via Modbus/SunSpec
- [ ] **Grid Stability Monitoring** - Detect coordinated attacks that could destabilize the power grid (simultaneous load manipulation, frequency injection)
- [ ] **DERMS Protection** - Secure Distributed Energy Resource Management Systems from unauthorized aggregation commands that could cause blackouts

#### Substation & Facility Physical Security
- [ ] **Perimeter Intrusion Detection** - Camera + fence sensor integration. Detect cutting, climbing, or tunneling at substations. Cross-camera Re-ID tracking of intruder
- [ ] **Transformer Monitoring** - Camera-based detection of physical attacks on transformers (shooting, arson). Combined with temperature/vibration sensors
- [ ] **Drone Surveillance** - Enhanced drone detection for critical infrastructure. Power lines, substations, and generation facilities are high-value drone targets
- [ ] **Copper Theft Detection** - Detect unauthorized vehicles, cutting tools, or humans near cable runs and transformer yards during off-hours
- [ ] **Environmental Compliance Camera** - Monitor for environmental violations (spills, emissions) with visual AI. Auto-document for EPA/regulatory reporting

#### NERC CIP Compliance Automation
- [ ] **CIP-002 Asset Identification** - Auto-inventory and classify all Bulk Electric System (BES) cyber assets. Maintain required asset lists
- [ ] **CIP-005 Electronic Security Perimeter** - Continuous monitoring of Electronic Security Perimeter boundaries. Detect unauthorized connections, verify firewall rules
- [ ] **CIP-007 System Security Management** - Track patches, ports/services, malware prevention, and security event monitoring per NERC requirements
- [ ] **CIP-010 Configuration Management** - Baseline configuration tracking for all BES assets. Auto-detect and alert on unauthorized changes. Maintain 35-day change records
- [ ] **CIP-011 Information Protection** - Track access to BES Cyber System Information. Enforce classification, storage, and disposal requirements
- [ ] **Evidence Package Generator** - Auto-compile NERC CIP audit evidence packages from monitoring data. Each control mapped to evidence with timestamps

### Real-Time Geographic Tracking & Prediction
Track targets across cameras with map overlay and predict movement when they leave camera coverage.

#### Geo-Mapped Camera System
- [ ] **Camera GPS Registration** - Each camera tagged with exact GPS coordinates, field-of-view angle, and range. System knows exactly what geographic area each camera covers
- [ ] **Live Map Overlay** - Real-time target positions plotted on a map (OpenStreetMap for offline/free, Google Maps API for online). Shows which camera is actively tracking the target
- [ ] **Camera Coverage Heatmap** - Visual map showing covered vs. blind spot areas. Helps clients identify where they need additional cameras
- [ ] **Dead Zone Mapping** - System knows where cameras can't see. When a target enters a dead zone, it calculates which camera should pick them up next based on direction of travel

#### Predictive Tracking
- [ ] **Direction-of-Travel Prediction** - When target exits camera view, AI calculates trajectory based on last known speed and heading. "Subject was heading northeast at walking pace, likely heading toward Oak Street"
- [ ] **Street-Level Prediction** - Using downloaded street map data (OpenStreetMap), predict which streets/intersections the target will hit based on available paths from last known position
- [ ] **Vehicle Route Prediction** - For tracked vehicles, predict likely route using road network data. "Vehicle heading south on I-95, next camera coverage at Exit 47 in approximately 3 minutes"
- [ ] **ETA to Next Camera** - Calculate estimated time of arrival at the next camera in the coverage area based on speed and predicted route. Alert operator: "Subject should appear on Camera 12 in approximately 45 seconds"
- [ ] **Historical Path Learning** - AI learns common movement patterns over time. "People leaving Building A at 5 PM typically walk to Parking Lot C." Use learned patterns to improve predictions

#### Multi-Channel Real-Time Notifications
Alert the right people through every channel simultaneously:
- [ ] **Email Alerts** - Rich HTML emails with embedded map screenshots, target photos, incident details. Configurable per-user, per-alert-type
- [ ] **SMS/Text Alerts** - Immediate text messages for critical events. Include GPS coordinates, target description, camera ID. Works with Twilio, AWS SNS, or direct carrier APIs
- [ ] **Automated Phone Calls** - Voice call alerts for highest-priority events (active shooter, perimeter breach, vault tampering). Text-to-speech reads the alert. Requires acknowledgment (press 1 to confirm)
- [ ] **Push Notifications** - Mobile app push notifications with target photo and live map link. One tap opens the live tracking view
- [ ] **Webhook Integration** - Fire webhooks to any external system (Slack, Teams, PagerDuty, ServiceNow, IFTTT). JSON payload with full incident data
- [ ] **Live Dashboard Feed** - Real-time event stream on the iTaK Shield dashboard. Map updates every second showing target movement
- [ ] **Escalation Chains** - If first responder doesn't acknowledge in X minutes, auto-escalate to next person. Keep escalating until someone responds
- [ ] **Alert Grouping** - Related alerts grouped into incidents. Don't send 47 separate "motion detected" alerts - send one "ongoing perimeter breach incident" with a live tracking link

#### Geographic Intelligence
- [ ] **Offline Map Support** - Download and cache OpenStreetMap tiles for offline operation. Critical for air-gapped and remote sites
- [ ] **Geofencing** - Draw virtual boundaries on the map. Alert when tracked target enters or exits a geofence zone
- [ ] **Multi-Site Map View** - Single map showing all client sites with real-time status. Zoom from global overview into individual camera feeds
- [ ] **BOLO Broadcasting** - Be On the Lookout: broadcast target description (Re-ID features, plate number, vehicle description) to all cameras across all sites. Every camera actively searching for the target
- [ ] **Law Enforcement Handoff Package** - One-click export: target photos from every camera, movement timeline, predicted direction, map with last known position, all video clips. Ready to hand to police on arrival

### iTaK Shield Voice Agent (TTS)
The iTaK Shield agent speaks. Customizable voice for automated calls, PA announcements, and real-time narration of incidents.

#### Voice Configuration
- [ ] **Custom Voice Setup** - Choose from multiple TTS voices (male/female, various accents, tones). Set different voices per alert type: calm voice for routine, urgent voice for critical
- [ ] **Voice Cloning** - Clone a specific person's voice (security director, facility manager) so alerts sound like they're coming from a recognized authority. Uses iTaK Torch local TTS inference
- [ ] **Multi-Language TTS** - Generate alerts in English, Spanish, French, Mandarin, etc. Auto-detect recipient's preferred language
- [ ] **Local TTS Engine** - All voice synthesis runs on-device via iTaK Torch. No cloud API calls, no data leaving the network. Works in air-gapped environments
- [ ] **Voice Profiles** - Save multiple voice profiles per deployment. "Professional" for client-facing, "Military" for government, "Calm" for healthcare

#### Automated Voice Actions
- [ ] **PA System Announcements** - iTaK Shield speaks over building PA system. "Attention: unauthorized person detected in Parking Lot B. Security to Parking Lot B immediately"
- [ ] **Phone Call Alerts** - AI-generated voice calls to security personnel, facility managers, or law enforcement. Reads incident details, requests acknowledgment
- [ ] **Two-Way Voice** - Speak to intruders through camera-connected speakers. "You are being recorded. Security has been dispatched to your location"
- [ ] **Incident Narration** - Real-time voice narration of tracked incidents for dispatchers: "Subject is now moving south through the lobby toward the east exit. Wearing dark jacket, blue jeans"
- [ ] **Voice Command Interface** - Security operators can issue voice commands to iTaK Shield: "Track that person," "Lock down Building C," "Show me Camera 12"

### Messaging Platform Integration
Send alerts through every platform people actually use:

#### Direct Messaging
- [ ] **WhatsApp Business API** - Send alerts via WhatsApp with photos, videos, maps, and voice messages. Supports WhatsApp voice calls for critical alerts
- [ ] **Signal Integration** - End-to-end encrypted alerts via Signal for maximum security. Text, images, and files
- [ ] **Telegram Bot** - Dedicated iTaK Shield Telegram bot per deployment. Real-time alerts with inline buttons for acknowledgment
- [ ] **SMS/MMS** - Traditional text messages with attached photos via Twilio, Vonage, or AWS SNS. Fallback when internet-based messaging is unavailable
- [ ] **Microsoft Teams** - Post alerts to Teams channels with adaptive cards. Click-through to live camera feeds
- [ ] **Slack** - Webhook-based alerts to Slack channels. Rich formatting with target photos, maps, and action buttons
- [ ] **Discord** - Webhook alerts for organizations using Discord for communications

#### Voice Calling Platforms
- [ ] **WhatsApp Voice Calls** - Automated voice calls through WhatsApp using iTaK Shield's TTS voice
- [ ] **SIP/VoIP Calling** - Direct SIP trunk integration for automated phone calls to any number. No third-party service needed if client has SIP infrastructure
- [ ] **Twilio Voice** - Automated phone calls via Twilio API. Programmable IVR: "Press 1 to acknowledge, Press 2 for more details, Press 3 to dispatch police"
- [ ] **PBX Integration** - Integrate with client's existing phone system (Cisco, Avaya, FreePBX). Ring security desk phone directly with incident details

### Emergency Dispatch Integration
Connect directly to 911 centers and law enforcement.

#### RapidSOS Integration (911 Direct)
RapidSOS connects to 4,800+ 911 centers (PSAPs) across the USA. Their API lets iTaK Shield send data directly to police/fire/EMS dispatch.
- [ ] **RapidSOS 911 API** - Send automated digital dispatch requests directly to 911 centers via RapidSOS API. Pre-populate dispatch with: GPS coordinates, incident type, target description, camera photos
- [ ] **Video-to-911** - Stream live camera feeds directly to the 911 dispatcher via RapidSOS. Dispatcher sees what iTaK Shield sees in real-time
- [ ] **Photo Transfer** - Send target photos, license plate captures, and weapon detection screenshots directly to responding officers' mobile terminals
- [ ] **Incident Data Package** - Automatically transmit full incident data to CAD (Computer-Aided Dispatch) systems: location, timeline, target movement history, severity classification
- [ ] **RapidSOS Sandbox** - Development sandbox environment for testing 911 integration without triggering real dispatch

#### Direct Law Enforcement Integration
- [ ] **CAD System Integration** - Connect to police department CAD systems (Motorola Solutions, Tyler Technologies, Hexagon). Push alerts directly into dispatch queue
- [ ] **NIBRS Reporting** - Auto-format incident reports per FBI's National Incident-Based Reporting System standard for law enforcement
- [ ] **Blue Alert Integration** - Receive and act on Blue Alerts (law enforcement officers in danger), AMBER Alerts, and Silver Alerts. Auto-activate BOLO scanning on all cameras
- [ ] **License Plate Database Query** - Query NCIC (National Crime Information Center) and state DMV databases for plate lookups (requires law enforcement authorization)
- [ ] **ShotSpotter/SoundThinking Integration** - Correlate iTaK Shield's gunshot detection with municipal ShotSpotter/SoundThinking acoustic sensor networks for precise triangulation
- [ ] **Real-Time Crime Center Feed** - Push iTaK Shield's live data directly to municipal Real-Time Crime Centers (RTCCs). Many major cities (NYC, Chicago, Houston, Atlanta) operate RTCCs that aggregate security feeds

#### Private Security Dispatch
- [ ] **Guard Tour Verification** - Camera Re-ID verifies security guards complete patrol routes on schedule. Alert if guard doesn't appear at checkpoint
- [ ] **Mobile Guard App** - Security guards receive alerts, acknowledge incidents, upload photos, and check in via mobile app
- [ ] **SOC Ticketing** - Every alert creates a ticket. Track response time, resolution, and follow-up. Audit trail for every incident

## 27. Additional Enhancements

### 1. Expanded Agent Personas and Industry-Specific Skills
- [ ] **Cryptocurrency/Blockchain Persona** - Wallet management, transaction monitoring, DeFi analytics, smart contract auditing. *[Operator + Researcher + crypto skills]* *(Enhancement)*
- [ ] **E-commerce Persona** - Product catalog management, inventory syncing (Shopify, WooCommerce), customer review analysis, automated pricing. *[Browser + Operator + e-commerce skills]* *(Enhancement)*
- [ ] **Agriculture Persona** - Crop monitoring via IoT sensors, weather-based alerts, supply chain tracking. *[Operator + Researcher + agriculture skills]* *(Enhancement)*
- [ ] **Telecommunications Persona** - Network diagnostics, customer support automation, billing dispute resolution. *[Operator + Researcher + telecom skills]* *(Enhancement)*
- [ ] **AI-Driven Personalization** - Adapt personas based on user behavior patterns and preferences. *[Researcher + personalization skills]* *(Enhancement)*

### 2. Advanced Multi-Modal and Sensory Integration
- [ ] **Haptic Feedback Integration** - Provide tactile responses (e.g., vibration on task completion) for mobile/desktop agents. *[Operator + haptic skills]* *(Enhancement)*
- [ ] **Environmental Sensor Connection** - Connect to smart home devices for context-aware actions (e.g., adjust lighting based on agent activity). *[Operator + smart home skills]* *(Enhancement)*
- [ ] **Real-Time Audio Processing** - Live audio analysis for sentiment, language detection, or noise filtering beyond transcription. *[Researcher + audio skills]* *(Enhancement)*
- [ ] **AR/VR Environment Support** - Expand iTaK Vision for immersive agent interactions in AR/VR. *[Browser + Vision + AR/VR skills]* *(Enhancement)*

### 3. Enhanced Scalability and Resource Management
- [ ] **Dynamic Resource Allocation** - Auto-scale worker agents based on task complexity and hardware availability (e.g., spin up GPU-accelerated workers for vision tasks). *[Operator + scaling skills]* *(Enhancement)*
- [ ] **Agent Federation** - Allow multiple iTaK Agent instances to collaborate across networks or clouds, with load balancing and failover. *[Operator + federation skills]* *(Enhancement)*
- [ ] **Energy-Efficient Modes** - Optimize for low-power devices (e.g., Raspberry Pi) with model quantization and batch processing. *[Operator + efficiency skills]* *(Enhancement)*

### 4. Improved User Experience and Accessibility
- [ ] **Voice-First Interfaces** - Full voice command support for hands-free operation, with natural language processing for complex queries. *[Researcher + voice skills]* *(Enhancement)*
- [ ] **Gamification in iTaK Teach** - Add badges, leaderboards, and progress tracking to make learning engaging. *[Researcher + gamification skills]* *(Enhancement)*
- [ ] **Universal Accessibility Compliance** - Ensure all UI components meet WCAG 2.2 AA, with screen reader support and customizable themes. *[Coder + accessibility skills]* *(Enhancement)*

### 5. Strengthened Security and Compliance
- [ ] **Zero-Knowledge Proofs** - Process data without storing or exposing it for enhanced privacy. *[Operator + security skills]* *(Enhancement)*
- [ ] **Compliance Automation** - Auto-generate reports for GDPR, HIPAA, or SOC 2, with agent-driven audits. *[Researcher + compliance skills]* *(Enhancement)*
- [ ] **AI-Based Threat Detection** - Use ML models to identify anomalous behavior in skills before execution. *[Doctor + AI skills]* *(Enhancement)*

### 6. Ecosystem and Community Features
- [ ] **Agent Marketplace Expansion** - Allow user-generated agents to be shared, rated, and monetized beyond iTaK Hub. *[Builder + marketplace skills]* *(Enhancement)*
- [ ] **Collaborative Workspaces** - Multi-user editing of agent workflows, with real-time syncing and conflict resolution. *[Operator + collaboration skills]* *(Enhancement)*
- [ ] **Open-Source Community Integration** - Direct connections to GitHub/GitLab for issue tracking, PR reviews, and code contributions. *[Coder + git skills]* *(Enhancement)*

### 7. Performance and Observability Upgrades
- [ ] **Predictive Analytics** - Use historical data to forecast agent performance, suggest optimizations, and prevent bottlenecks. *[Researcher + analytics skills]* *(Enhancement)*
- [ ] **Real-Time Debugging Tools** - Step-through execution for agent workflows, with breakpoints and variable inspection. *[Coder + debugging skills]* *(Enhancement)*
- [ ] **Carbon Footprint Tracking** - Track and optimize the environmental impact of AI operations. *[Researcher + sustainability skills]* *(Enhancement)*

### 8. Future-Proofing with Emerging Tech
- [ ] **Quantum-Resistant Encryption** - Implement encryption methods resistant to quantum computing threats. *[Operator + encryption skills]* *(Enhancement)*
- [ ] **Decentralized Protocols** - Support IPFS for knowledge graphs and blockchain for immutable logs. *[Operator + decentralized skills]* *(Enhancement)*
- [ ] **Edge Computing Support** - Run lightweight agents on IoT devices for distributed intelligence. *[Operator + edge skills]* *(Enhancement)*
