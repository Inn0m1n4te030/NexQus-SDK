# Security and support

This candidate supports local simulation of Python-defined circuits. It is not a
sandbox, a hosted execution service, a cryptographic random number generator or
a hardware security product. Notebook execution runs ordinary Python code.
Do not run untrusted scripts or notebooks in an account containing credentials.

The owner must configure private security reporting on the verified source
repository before public release. No unverified email address is supplied here.
Use that private reporting channel for a security-sensitive finding; ordinary
correctness bugs belong in https://github.com/Inn0m1n4te030/NexQus-SDK/issues.

After v1.0 is published, the latest 1.x patch release is the supported release
line. Earlier 0.x prototypes have no promised security maintenance. The project
does not promise response-time SLAs or verified compatibility with every future
NumPy, Torch, CUDA or Python release.

Memory checks estimate quantum working tensors; they do not enforce a process
RAM/VRAM cap or bound arbitrary Python circuit construction. Applications that
execute untrusted requests need their own isolation, resource limits and input
limits outside the SDK.

Dependency and publishing credentials do not belong in source, notebooks,
examples, reports or issue attachments. The publishing workflow uses PyPI
Trusted Publishing and does not require long-lived PyPI tokens in this repository.
