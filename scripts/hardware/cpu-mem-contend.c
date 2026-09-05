// Stream host memory on N threads to contend for the shared LPDDR5X.
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

static volatile int stop = 0;
static size_t per = (size_t)1 << 30;  // 1 GiB per thread

void* work(void* arg) {
  (void)arg;
  char* a = malloc(per);
  char* b = malloc(per);
  if (!a || !b) { fprintf(stderr, "alloc fail\n"); return NULL; }
  memset(a, 1, per);
  memset(b, 2, per);
  while (!stop) memcpy(b, a, per);
  free(a); free(b);
  return NULL;
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 8;
  int secs = (argc > 2) ? atoi(argv[2]) : 30;
  pthread_t t[64];
  for (int i = 0; i < n; i++) pthread_create(&t[i], NULL, work, NULL);
  sleep(secs);
  stop = 1;
  for (int i = 0; i < n; i++) pthread_join(t[i], NULL);
  return 0;
}
