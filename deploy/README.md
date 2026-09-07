# Public V2 deployment

The public release is an anonymous, read-only search demo over the bundled educational
corpus. The local application retains PDF/TXT/Markdown import and library management.
Do not expose a local library or mount private documents into the public container.

## Runtime

Build from the repository root:

```sh
docker build -t bahith-v2 .
docker run --rm -p 127.0.0.1:7860:7860 \
  -e BAHITH_ALLOWED_HOSTS=localhost,127.0.0.1 \
  bahith-v2
```

For an internet deployment, use the hosting provider's HTTPS endpoint and set
`BAHITH_ALLOWED_HOSTS` to its exact public hostname, without scheme, port, path or a wildcard.
`PORT` defaults to 7860 and may be changed by the provider. The container runs one worker
as a non-root user and explicitly installs CPU PyTorch. It excludes local databases,
uploads, model caches, Git history and evaluation files from its build context.

The model revision is pinned in `search.py`. Model files are downloaded from Hugging Face
at initial startup. A normal restart can download them again on ephemeral hosts, unless
the provider offers a cache volume. No inference API key is required for this public model.

Public defaults:

- `BAHITH_PUBLIC_DEMO=1`: read-only public behavior with a clean temporary demo library.
- `BAHITH_PRELOAD_MODEL=1`: prepare the model and passage embeddings before readiness.
- `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`: bounded CPU parallelism; tune only after measuring.
- `BAHITH_ALLOWED_HOSTS`: required by the container launcher; use exact hosting names.
- `BAHITH_ALLOW_HF_EMBED=1`: optional when using the Hugging Face page wrapper; permits
  only its official frame ancestors. The default denies framing. The assigned direct
  `.hf.space` app URL works without enabling this option.

`/health` is a liveness check. Use `/ready` for readiness; do not announce a working search
demo until this endpoint returns 200 and a real semantic query succeeds through public HTTPS.
Keep the host at one worker: the application uses process-local admission and caching.
Cold startup includes encoding the demo passages and can take several minutes on CPU.
Configure a bounded startup timeout and restart policy on the selected host; Docker's
health check alone does not restart an unhealthy container. Preload failures are logged
for operators, while public readiness responses omit the underlying error details.
Provider-level request limits and TLS remain necessary. This is a bounded public demo,
not a multi-user document-management service or a traffic-capacity guarantee.

## Hugging Face Docker Spaces

Use an authenticated account you own. Create a public Docker Space on CPU Basic only
when that hardware is already included in your account; do not select paid upgrades
without a budget decision. Upload the application files and Dockerfile, using
`deploy/SPACE_README.md` as the Space's root `README.md`. Copy the assigned app hostname
from Hugging Face into `BAHITH_ALLOWED_HOSTS` in the Space variables. Never guess the URL.

As checked on 2026-09-07, the official overview says creating a new Docker Space requires
a paid account plan, even though CPU Basic itself has no hourly charge. Account eligibility
and current pricing must be checked before creation. This guide does not authorize a purchase.

- [Spaces overview and eligibility](https://huggingface.co/docs/hub/spaces-overview)
- [Docker Space configuration](https://huggingface.co/docs/hub/en/spaces-sdks-docker)

Any other container host with enough RAM can use the same image. The .6B embedding model
needs substantially more than a static-site host; confirm actual startup memory and latency
on the selected machine. GitHub Pages and the current Sites Worker runtime cannot directly
run this Python/PyTorch container.

## Release acceptance

1. All Python tests pass and the container builds with a valid dependency check.
2. Local behavior still supports import/replace/delete; the public mode rejects mutations.
3. Public pages expose only the seeded demo corpus, with no local files or secret tokens.
4. Invalid hosts are rejected, search overload receives a bounded response, and readiness
   tracks model/index preparation accurately.
5. Test one real semantic query and a dimension change through the deployed HTTPS URL;
   inspect the source view, Arabic layout, and public-mode notice on desktop and mobile.
6. Record the exact deployed commit, assigned URL, hardware and readiness result.

## Evidence for the launch

Use the metrics in `docs/benchmarks/README.md`: 96.7% Success@5, 90.0% P@1 and .9281 MRR@5
on 120 authored queries against 30 demo passages at 1024 dimensions. Always state that scope.
The synthetic fixture's perfect result is not independent evidence and should not headline
the launch. The 2.4ms local cached-dimension observation is not public end-to-end latency.

No live public URL is recorded here until a host has accepted and verified the deployment.

The pinned Arabic model card declares that it inherits its base model's license. The
[Microsoft base model](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) is marked MIT
(checked 2026-09-07). The container preserves the application's MIT notice; model weights
are fetched from the publisher at runtime and attribution is linked in the interface.
The public corpus is the existing MIT-licensed repository's 30 short educational examples,
clearly marked as demonstration content; no imported personal files are published.
