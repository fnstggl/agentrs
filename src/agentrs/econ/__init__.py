"""Multi-domain economic task environments, a generative tool-using policy, and the overnight
continual-RL loop (see docs/overnight.md).

Domains (all locally executable, independently verified by hidden verifiers):
  software   – sandboxed Python repos: bug fixes, API repairs, config/dependency fixes, implementation
  support    – τ²-style customer-support / account operations over SQLite with a scripted customer
               (a custom fallback environment, NOT τ²-Bench itself)
  browser    – the existing MiniWeb simulated web apps (forms, login, checkout, lookup)
  finance    – invoices, payments, expenses, ledgers, budgets over SQLite
  crm        – companies, contacts, leads, deals, proposals, meeting scheduling over SQLite
  research   – document collections with conflicting sources; attributable, numerically checked answers
"""
