# GitHub Pages frontend and independent model service

GitHub Pages serves the Arabic interface, bundled demo documents, passage/source views,
styles, and browser JavaScript. The Python model service performs actual retrieval and
Matryoshka processing. No query answers, scores, or query embeddings are precomputed into
the static site. The model name, revision, ranking defaults, and evaluation evidence remain
unchanged by this hosting split.

## Prepare the model service

Deploy the public Docker image using [the runtime guide](README.md). Set its exact assigned
hostname in `BAHITH_ALLOWED_HOSTS` and set:

```text
BAHITH_ALLOWED_ORIGINS=https://abdulrahman-s-asiri.github.io
BAHITH_PUBLIC_DEMO=1
BAHITH_PRELOAD_MODEL=1
```

Do not expose the personal local application. Verify `/ready` returns 200 through HTTPS
and a real semantic `/api/query` request succeeds before configuring the frontend.
The CORS setting grants this Pages origin read access without cookies or credentials.
All public mutation restrictions still apply. CORS is not authentication or a traffic limit.

## Build the static site

From the source repository with the model-free test dependencies installed:

```sh
python deploy/export_pages.py ../bahith-pages-build --base-path /bahith/ --api-base https://your-assigned-model-host.example
```

Replace the example URL with the verified service origin. The output must be a new or empty
directory. Omitting `--api-base` produces a browsable site that clearly says live search is
being prepared; it never simulates results. Never advertise an unconfigured site as a fully
working search deployment.

The exporter always creates a fresh public demo library; it does not import local documents,
read personal embeddings, or download a model. It exports only the known public routes and
static assets, with links under `/bahith/`. The browser renders JSON search responses with
text nodes and keeps source navigation inside the published demo library. Live search needs
JavaScript; the bundled pages remain browsable without it.

## Publish

Commit the generated output to the repository's dedicated `gh-pages` branch. Do not merge
unrelated application branches into `main` just to publish. Configure GitHub Pages to deploy
from `gh-pages` at `/` (the output includes `.nojekyll`). The expected project URL follows
GitHub's normal owner/repository format, but report only the actual `html_url` returned by
the Pages service after successful deployment.

For each deployment, record the source commit, generated branch commit, assigned model URL,
hardware, and readiness/search checks outside the source repository. Deploy frontend and
backend from the same source revision so the static passage IDs match the model corpus.

## Acceptance

- Python tests and static export pass; generated files contain no private data or runtime files.
- The GitHub Pages landing page, nested routes, and static assets load over HTTPS.
- A new Arabic question returns real results from the configured backend while the browser
  stays on the GitHub Pages origin. Test both 1024 and 512 dimensions and the source links.
- CORS rejects unconfigured origins; mutation endpoints still reject requests.
- Network failure shows an availability message and does not substitute lexical or canned results.
- No independent-service claim is made for a temporary tunnel running on the user's computer.

Provider reference: [GitHub Pages publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
