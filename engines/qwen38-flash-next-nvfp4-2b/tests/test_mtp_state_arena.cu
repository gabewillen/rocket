#include "mtp/state_arena.h"
#include <cstdlib>
int main(){rocket::qwen38::mtp::StateArena a(16,4,true);auto f=a.prefix(0),l=a.prefix(3);if(a.allocated_bytes()==0||!f.main_key||!f.main_value||!f.raw_key||!f.compressed_key||!f.rope_positions||!f.multi_hidden||f.multi_hidden==l.multi_hidden)std::abort();return 0;}
