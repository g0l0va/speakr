# Local smoke-test overlay (persistent-speakers diff)

Builds and runs the current checkout's changes on top of the **exact pinned
production base image** (`learnedmachine/speakr:lite@sha256:c4d5047252cbaa82987104e9e3378e205c1d4a6c9080a7f3326b021a4b944ed5`,
upstream revision `b2dc897677035321737eee07906d5ce0a2d8add2`), in a stack that
is completely isolated from the real pilot deployment.

**Not for production use. Never push the built image anywhere.**

## What's isolated

- Separate Docker network (`speakr-smoke-net`), separate container names
  (`speakr-smoke-app`, `speakr-smoke-db`) -- nothing shared with the pilot's
  `speakr-net` / `speakr-app` / `speakr-db`.
- Separate, **ephemeral** Postgres: no volume, so `down` (no `-v` needed)
  leaves zero data behind and the next `up` always starts from empty.
- Separate port: `127.0.0.1:8898` (the pilot uses `SPEAKR_PORT` from its own
  `compose/.env`, normally a different port).
- No ASR service. This smoke pass only exercises login, the account page
  (including the new voice-sample review UI), and DB migrations -- not
  transcription.

## What's copied vs. reused from the base image

`deployment/Dockerfile.local-overlay` copies only `src/`, `templates/`, and
`static/` from this checkout onto the pinned base image. Everything else the
base image already has -- OS packages, Python dependencies, ffmpeg, and the
downloaded `static/vendor/` JS/CSS/font assets -- is reused unchanged. See
the comments in that file for why this can't accidentally wipe out
`static/vendor/`.

## Running it

From the repo root (`services/speakr-app/`). No secrets needed -- every
credential in `docker-compose.smoke.yml` is a fixed, throwaway smoke-only
value, not a real one.

```bash
# Validate the compose file first
docker-compose -f deployment/docker-compose.smoke.yml config

# Build the image and start the stack
docker-compose -f deployment/docker-compose.smoke.yml up -d --build

# Watch both containers reach healthy
docker-compose -f deployment/docker-compose.smoke.yml ps

# Tear down when done (ephemeral DB has no volume -- nothing else to clean up)
docker-compose -f deployment/docker-compose.smoke.yml down
```

If your Docker daemon runs under a non-default context (e.g. this pilot's
`colima-speakr`), prefix every command with `DOCKER_CONTEXT=colima-speakr`,
same as the pilot's own scripts do.

## Smoke-test login

| | |
|---|---|
| URL | http://127.0.0.1:8898 |
| Email | `smoke-admin@example.com` |
| Password | `SmokeTest123!` |

The login form authenticates by email, not username -- `smoke_admin` is
only the internal username `docker_create_admin.py` stores for this
account; log in with the email above.

These are fixed, fictitious smoke-test-only credentials created fresh by
`scripts/docker_create_admin.py` on first boot of the ephemeral DB -- they
exist nowhere else and are not real secrets. The admin email
(`smoke-admin@example.com`) uses RFC 2606's documentation-only domain
deliberately: `.local`/`.internal`/`.test` are rejected outright by
email-validator's reserved-name check (not just the deliverability check
`SKIP_EMAIL_DOMAIN_CHECK` controls), so `example.com` -- syntactically valid,
just missing an MX record -- is the domain that actually gets past both
checks. Same finding already documented in `../docs/baseline.md` for the
pilot's own `ADMIN_EMAIL`.

## What was verified automatically (no browser)

- `docker-compose ... config` — the compose file parses and resolves cleanly.
- Both containers reach `healthy`.
- `POST /login` with the credentials above succeeds and returns an
  authenticated session.
- `GET /account` with that session returns 200 and contains the Speakers
  Management markup, including the new voice-sample review panel's source
  (confirmed present, not that it renders correctly in a browser).
- The database migration ran: `speaker_embedding_sample` exists with its
  expected columns, CHECK constraints, and the partial unique index, via a
  direct query against `speakr-smoke-db`.

## What's left for manual GUI QA

Everything that actually requires a browser: does the Speakers Management
tab render correctly, does the voice-sample review panel expand/collapse
and paint its table properly, do Confirm/Reject/Delete actually update the
UI without a reload, does the status filter work, does layout hold up
in both light and dark themes. The stack is left running after the
automated checks so this can be done independently. Point Speaker
Embedding samples aside, general login/account/recording flows other than
transcription (no ASR configured here) are also fair game to click through.
