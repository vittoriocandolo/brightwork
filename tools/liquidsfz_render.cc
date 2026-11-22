// Copyright (C) 2026 Vittorio Candolo
// SPDX-License-Identifier: GPL-3.0-or-later
//
// An offline renderer for liquidsfz, which ships only real-time clients, used by tests/test_sfz_player.py.
//
// Usage: liquidsfz_render SFZ EVENTS FRAMES RATE OUT
//   EVENTS has one "frame key velocity" per line, velocity 0 for a note off; OUT is raw float32, stereo interleaved.
//
// Build against a liquidsfz checkout (https://github.com/swesterfeld/liquidsfz) and libsndfile:
//   g++ -O2 -std=c++17 -I liquidsfz/lib tools/liquidsfz_render.cc liquidsfz/lib/*.cc -lsndfile -lpthread \
//       -o liquidsfz_render
// then point LIQUIDSFZ_RENDER at the binary.
#include "liquidsfz.hh"
#include <cstdio>
#include <fstream>
#include <vector>
#include <algorithm>

int main (int argc, char **argv)
{
  if (argc != 6) { fprintf (stderr, "usage: render SFZ EVENTS FRAMES RATE OUT\n"); return 2; }
  LiquidSFZ::Synth synth;
  synth.set_sample_rate (atoi (argv[4]));
  synth.set_live_mode (false);
  if (!synth.load (argv[1])) { fprintf (stderr, "cannot load %s\n", argv[1]); return 1; }
  struct Event { long frame; int key, velocity; };
  std::vector<Event> events;
  std::ifstream in (argv[2]);
  long f; int k, v;
  while (in >> f >> k >> v) events.push_back ({f, k, v});
  std::sort (events.begin(), events.end(), [] (const Event& a, const Event& b) { return a.frame < b.frame; });
  long total = atol (argv[3]);
  FILE *out = fopen (argv[5], "wb");
  const long block = 256;
  std::vector<float> left (block), right (block), inter (2 * block);
  size_t next = 0;
  for (long pos = 0; pos < total; pos += block)
    {
      long n = std::min (block, total - pos);
      while (next < events.size() && events[next].frame < pos + n)
        {
          const Event& e = events[next++];
          if (e.velocity > 0)
            synth.add_event_note_on (e.frame - pos, 0, e.key, e.velocity);
          else
            synth.add_event_note_off (e.frame - pos, 0, e.key);
        }
      float *outputs[2] = { left.data(), right.data() };
      synth.process (outputs, n);
      for (long i = 0; i < n; i++) { inter[2 * i] = left[i]; inter[2 * i + 1] = right[i]; }
      fwrite (inter.data(), sizeof (float), 2 * n, out);
    }
  fclose (out);
  return 0;
}
