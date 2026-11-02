# L: Llama 3.2 1B Instruct replication: BLOCKED at L1 (gated model, no credentials)

Pre-registration: [`preregistration.md`](preregistration.md) (`9c0f3b0`). The L1 load check could not download
`meta-llama/Llama-3.2-1B-Instruct`: the repository is gated and this machine has no Hugging Face token ([`load_check.json`](load_check.json)).
Per the pre-registration, L stops here. L2/L3 were not run, so the Llama question is **untested**.

To unblock: accept the Llama 3.2 license on Hugging Face with the account owner's login, run `hf auth login` locally, and rerun L1.
Ungated community mirrors of the same weights exist. Using one would be a deviation from the pre-registered model ID and needs an explicit decision.
