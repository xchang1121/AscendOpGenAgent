"""utils/ — stateless library modules imported by engine/, hooks/,
phase_machine/, workflow/, task_config/, and batch/.

No CLI entry points live here. `validate_triton_impl.py` is a thin
re-export of the kernel-verifier skill's canonical implementation —
`from utils.validate_triton_impl import validate` resolves through to
`skills/triton/kernel-verifier/scripts/validate_triton_impl.py` so the
two consumers (autoresearch + the skill itself) can never drift.

Nothing in this package mutates state. Splitting them out makes the
dependency direction obvious: utils sits at the bottom of the stack and
never imports from any sibling package.

---------------------------------------------------------------------------
Invariant — IMPORT STYLE INSIDE utils/

When a module in utils/ imports another utils/ module, use the
relative form: `from .settings import …`, NOT the absolute
`from settings import …`. The absolute form silently relies on
`scripts/utils/` being in sys.path — daemons (worker, batch driver)
add only `scripts/`, so the absolute form works in ad-hoc CLI runs
but blows up the first time a long-running process imports the
module.

If you're adding a new utils module that needs another utils module:
always use `from .<sibling> import X`. Audit your callsite for the
absolute form before pushing.
---------------------------------------------------------------------------
"""
