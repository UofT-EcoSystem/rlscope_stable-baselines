import textwrap

import iml_profiler.api as iml

def before_each_iteration(iteration, num_iterations):
  if iml.prof.delay and iml_is_warmed_up() and not iml.prof.tracing_enabled:
    # Entire training loop is now running; enable IML tracing
    iml.prof.enable_tracing()

  # GOAL: we only want to call report_progress once we've seen ALL the operations run
  # (i.e., q_backward, q_update_target_network).  This will ensure that the GPU HW sampler
  # will see ALL the possible GPU operations.
  if iml.prof.debug:
    iml.logger.info(textwrap.dedent("""\
        RLS: @ t={iteration}: OPERATIONS_SEEN = {OPERATIONS_SEEN}
          waiting for = {waiting_for}
        """.format(
      iteration=iteration,
      OPERATIONS_SEEN=OPERATIONS_SEEN,
      waiting_for=OPERATIONS_AVAILABLE.difference(OPERATIONS_SEEN),
    )).rstrip())
  if OPERATIONS_SEEN == OPERATIONS_AVAILABLE:
    OPERATIONS_SEEN.clear()
    iml.prof.report_progress(
      percent_complete=iteration/float(num_iterations),
      num_timesteps=iteration,
      total_timesteps=num_iterations)

    # if FLAGS.log_stacktrace_freq is not None and iteration % FLAGS.log_stacktrace_freq == 0:
    #   log_stacktraces()

OPERATIONS_SEEN = set()
OPERATIONS_AVAILABLE = set()

def iml_register_operations(operations):
  for operation in operations:
    OPERATIONS_AVAILABLE.add(operation)

def iml_prof_operation(operation):
  should_skip = operation not in OPERATIONS_AVAILABLE
  op = iml.prof.operation(operation, skip=should_skip)
  if not should_skip:
    OPERATIONS_SEEN.add(operation)
  return op

def iml_is_warmed_up():
  """
  Return true once we are executing the full training-loop.

  :return:
  """
  assert OPERATIONS_SEEN.issubset(OPERATIONS_AVAILABLE)
  # can_sample = replay_buffer.can_sample(batch_size)
  # return can_sample and OPERATIONS_SEEN == OPERATIONS_AVAILABLE and num_timesteps > learning_starts
  return OPERATIONS_SEEN == OPERATIONS_AVAILABLE
