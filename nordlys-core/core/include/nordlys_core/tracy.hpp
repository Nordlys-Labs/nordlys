#pragma once

// Tracy profiler integration for nordlys-core
//
// Usage:
//   - NORDLYS_ZONE: Mark entire function for profiling
//   - NORDLYS_ZONE_N(name): Mark function with custom name
//   - NORDLYS_ZONE_SCOPED(name): Mark scope with custom name
//   - NORDLYS_FRAME_MARK: Mark frame boundary (for benchmarks)
//
// When TRACY_ENABLE is not defined, all macros are no-ops (zero overhead).

#ifdef TRACY_ENABLE

#  include <tracy/Tracy.hpp>

#  define NORDLYS_ZONE ZoneScoped
#  define NORDLYS_ZONE_N(name) ZoneScopedN(name)
#  define NORDLYS_ZONE_SCOPED(name) ZoneScopedNC(name, tracy::Color::Blue)
#  define NORDLYS_FRAME_MARK FrameMark

#else

// No-op macros when Tracy is disabled (zero overhead)
#  define NORDLYS_ZONE
#  define NORDLYS_ZONE_N(name)
#  define NORDLYS_ZONE_SCOPED(name) (void)0
#  define NORDLYS_FRAME_MARK

#endif
