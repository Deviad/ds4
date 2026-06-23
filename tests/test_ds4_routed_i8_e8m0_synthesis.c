/* Story 11.49 (ADR 0021) loader paired-tensor synthesis unit test.
 *
 * Validates the post-load synthesis pass that flips routed-expert weight tensors
 * from raw i8 (gguf_type 24) to DS4_TENSOR_I8_E8M0 (64) when a sibling .scale
 * tensor is present, plus the type-64 routed-expert helper values.
 *
 * ds4_model / ds4_tensor are defined only in the ds4.c TU, so the in-memory
 * model is built there behind two non-static test hooks (mirrors the Story 11.47
 * ds4_gpu_test_i8_e8m0_host_dispatch_routing convention). This file is #included
 * by tests/ds4_test.c so it reuses the tiny TEST_ASSERT harness.
 *
 * Coverage (architecture.md §2 a-f):
 *   (a) raw-i8 weight + .scale sibling -> type 24 -> 64.
 *   (b) tensor_is_routed_expert_type(64) == true.
 *   (c) routed_expert_block_bytes(64) == 1.
 *   (d) routed_expert_row_bytes(dim[0]=2048,type=64) == 2048 (NOT the QK_K 8).
 *   (e) raw-i8 weight WITHOUT a .scale sibling -> ds4_die (forked child, exit!=0).
 *   (f) Q4_K weight -> no-op (type unchanged).
 */

#include <sys/wait.h>

extern int  ds4_test_routed_i8_e8m0_synthesis(void);
extern void ds4_test_routed_i8_e8m0_missing_sibling(void);

static void test_routed_i8_e8m0_synthesis(void) {
    /* (a)-(d),(f) all asserted inside the in-TU hook; rc==0 means every check passed. */
    const int rc = ds4_test_routed_i8_e8m0_synthesis();
    if (rc != 0) {
        fprintf(stderr, "ds4-test: routed-i8-e8m0-synthesis hook failed rc=%d\n", rc);
    }
    TEST_ASSERT(rc == 0);

    /* (e) fail-closed: missing .scale sibling must ds4_die (exit(1)). Fork so the
     * intended process exit does not tear down the parent test runner. */
    pid_t pid = fork();
    TEST_ASSERT(pid >= 0);
    if (pid == 0) {
        /* Child: silence stderr noise from the expected die, then trigger it. */
        ds4_test_routed_i8_e8m0_missing_sibling();
        _exit(0); /* unreachable on correct code: the helper must have died. */
    }
    int status = 0;
    TEST_ASSERT(waitpid(pid, &status, 0) == pid);
    /* Correct fail-closed behavior: ds4_die -> exit(1) (non-zero, normal exit). */
    TEST_ASSERT(WIFEXITED(status));
    TEST_ASSERT(WEXITSTATUS(status) != 0);
}
