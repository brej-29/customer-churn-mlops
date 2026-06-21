# Retention Playbook Corpus

> **Note:** These entries are illustrative demo tactics assembled for the RAG prototype.
> They represent common industry best-practices, **not** a real company playbook or
> proprietary customer data. Do not use them verbatim in production communications.

The corpus is indexed by `churn/genai/rag.py` using FAISS + `sentence-transformers/all-MiniLM-L6-v2`.
Each `.md` file covers one churn-risk signal and contains 2–3 concrete retention tactics.

| File | Churn Signal |
|------|-------------|
| `contract_upgrade.md` | Month-to-month contract |
| `new_customer_onboarding.md` | Short tenure / new customer |
| `price_sensitivity.md` | High monthly charges |
| `security_addon.md` | Missing OnlineSecurity / TechSupport |
| `fiber_quality.md` | Fiber-optic service quality |
| `autopay_incentive.md` | Electronic-check payment |
| `paperless_engagement.md` | Paperless billing adoption |
| `senior_retention.md` | Senior customer / no dependents |
