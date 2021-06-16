#include "../dep/tl/include/tl/common.h"
using namespace TL;

#include "../dep/tl/include/tl/window.h"
#include "../dep/tl/include/tl/console.h"
#include "../dep/tl/include/tl/opengl.h"
#include "../dep/tl/include/tl/math_random.h"
#include "../dep/tl/include/tl/mesh.h"

umm get_hash(v3s const &);
umm get_hash(struct RemeshRequest const &);

#include "../dep/tl/include/tl/hash_map.h"
#include "../dep/tl/include/tl/hash_set.h"

using namespace OpenGL;

#define for_v3(v3s, local, start, comp, end, inc) \
	for (v3s local=start;local.z comp end.z;local.z inc) \
	for (      local.y=0;local.y comp end.y;local.y inc) \
	for (      local.x=0;local.x comp end.x;local.x inc)

Window *window;
v3f camera_position = {0,0,0};
v3f camera_rotation = {};
f32 frame_time = 1.0f / 60;
PreciseTimer frame_timer;
f32 time;

GLuint surface_shader;
GLuint wireframe_shader;

u32 const grid_resolution = 32;

#define DEBUG_VERTEX 0

#if DEBUG_VERTEX
struct DebugVertex {
	f32 debug_distance;
};
#endif

struct Vertex {
	v3f position;
	v3f normal;
#if DEBUG_VERTEX
	DebugVertex debug;
#endif
};

inline static constexpr auto XXX = sizeof(Vertex);

using DistanceCell = s8;

#if 0

// Four-bit cells
//

inline static constexpr s32 distance_cell_min = 0;
inline static constexpr s32 distance_cell_max = 15;
inline static constexpr s32 distance_cell_surface = (distance_cell_min + distance_cell_max) / 2;

struct DistanceGrid {
	u8 data[grid_resolution * grid_resolution * grid_resolution / 2] = {};

	DistanceCell get(u32 x, u32 y, u32 z) {
		u8 result = data[(z * grid_resolution * grid_resolution + y * grid_resolution + x) / 2];
		if (x & 1) result >>= 4;
		else result &= 0xf;
		return result;
	}
	void set(u32 x, u32 y, u32 z, DistanceCell value) {
		if (value == distance_cell_surface)
			++value;
		u32 index = (z * grid_resolution * grid_resolution + y * grid_resolution + x) / 2;
		u8 dest = data[index];
		if (x & 1) dest = (dest & 0x0f) | ((value & 0xf) << 4);
		else       dest = (dest & 0xf0) | ((value & 0xf) << 0);
		data[index] = dest;
	}

	void set_normalized(u32 x, u32 y, u32 z, f32 value) {
		set(x, y, z, map<f32>(value, -1, 1, distance_cell_min, distance_cell_max));
	}
	static f32 get_t(u8 a, u8 b) {
		return (f32)((s32)a - (s32)distance_cell_surface) / ((s32)a - (s32)b);
	}
};
#else
//
// Eight-bit cells
//
inline static constexpr s32 distance_cell_min = -128;
inline static constexpr s32 distance_cell_max = 127;
inline static constexpr s32 distance_cell_surface = (distance_cell_min + distance_cell_max) / 2;
#endif

enum GridIndex {
	GridIndex_self,
	GridIndex_x,
	GridIndex_y,
	GridIndex_z,
	GridIndex_xy,
	GridIndex_xz,
	GridIndex_yz,
	GridIndex_xyz,
	GridIndex_count,
};


struct Cell {
	s8 distance;
#if DEBUG_VERTEX
	f32 debug_distance;
#endif
};

using Grid = Array3<Cell, grid_resolution, grid_resolution, grid_resolution>;

struct Chunk {
	v3s position = {};
	Grid grid;
	u32 index_count = 0;
	union {
		struct {
			GLuint vertex_buffer, index_buffer;
		};
		GLuint buffers[2] = {};
	};
	GLuint vertex_array = 0;
	u8 neighbor_count = 0;
	List<Vertex> vertices;
	List<u32> indices;
};

namespace MeshGenerator {

enum {
	 left_bottom_front,
	right_bottom_front,
		left_top_front,
	   right_top_front,
	  left_bottom_back,
	 right_bottom_back,
		 left_top_back,
		right_top_back,
};

struct Edge {
	u8 vertex_a;
	u8 vertex_b;
};

forceinline constexpr bool operator==(Edge a, Edge b) {
	return a.vertex_a == b.vertex_a && a.vertex_b == b.vertex_b;
}

static v3f base_positions[8];
static constexpr Edge edge_left_front   = {left_top_front,     left_bottom_front };
static constexpr Edge edge_left_back    = {left_top_back,      left_bottom_back  };
static constexpr Edge edge_right_front  = {right_top_front,    right_bottom_front};
static constexpr Edge edge_right_back   = {right_top_back,     right_bottom_back };
static constexpr Edge edge_left_bottom  = {left_bottom_front,  left_bottom_back  };
static constexpr Edge edge_left_top     = {left_top_front,     left_top_back     };
static constexpr Edge edge_right_bottom = {right_bottom_front, right_bottom_back };
static constexpr Edge edge_right_top    = {right_top_front,    right_top_back    };
static constexpr Edge edge_top_front    = {left_top_front,     right_top_front   };
static constexpr Edge edge_top_back     = {left_top_back,      right_top_back    };
static constexpr Edge edge_bottom_front = {left_bottom_front,  right_bottom_front};
static constexpr Edge edge_bottom_back  = {left_bottom_back,   right_bottom_back };

Array<StaticList<Edge, 12>, 256> intersected_edges; // TODO: can be constexpr

void init() {
	base_positions[left_bottom_back  ] = {0,0,0};
	base_positions[left_bottom_front ] = {0,0,1};
	base_positions[left_top_back     ] = {0,1,0};
	base_positions[left_top_front    ] = {0,1,1};
	base_positions[right_bottom_back ] = {1,0,0};
	base_positions[right_bottom_front] = {1,0,1};
	base_positions[right_top_back    ] = {1,1,0};
	base_positions[right_top_front   ] = {1,1,1};

	//
	// Setup intersected_edges:
	// For each case set which edges cross the surface
	//

	// Intersected edges are the same if signs are flipped
#define CASE(n) intersected_edges[n] = intersected_edges[(u8)~n] =

	// One corner
	CASE(0b00000001) { edge_left_bottom,  edge_left_front,   edge_bottom_front };
	CASE(0b00000010) { edge_right_bottom, edge_bottom_front, edge_right_front  };
	CASE(0b00000100) { edge_left_top,     edge_top_front,    edge_left_front   };
	CASE(0b00001000) { edge_right_top,    edge_right_front,  edge_top_front    };
	CASE(0b00010000) { edge_left_back,    edge_left_bottom,  edge_bottom_back  };
	CASE(0b00100000) { edge_right_bottom, edge_right_back,   edge_bottom_back  };
	CASE(0b01000000) { edge_left_back,    edge_top_back,     edge_left_top     };
	CASE(0b10000000) { edge_right_back,   edge_right_top,    edge_top_back     };

	// Two corners on the same side
	CASE(0b00001001) {edge_left_bottom,  edge_left_front,   edge_top_front,    edge_right_top,    edge_right_front,   edge_bottom_front};
	CASE(0b00000110) {edge_right_bottom, edge_bottom_front, edge_left_front,   edge_left_top,     edge_top_front,     edge_right_front };
	CASE(0b10010000) {edge_right_top,    edge_top_back,     edge_left_back,    edge_left_bottom,  edge_bottom_back,   edge_right_back  };
	CASE(0b01100000) {edge_left_top,     edge_left_back,    edge_bottom_back,  edge_right_bottom, edge_right_back,    edge_top_back    };
	CASE(0b01000001) {edge_top_back,     edge_left_top,     edge_left_front,   edge_bottom_front, edge_left_bottom,   edge_left_back   };
	CASE(0b00010100) {edge_bottom_back,  edge_left_back,    edge_left_top,     edge_top_front,    edge_left_front,    edge_left_bottom };
	CASE(0b10000010) {edge_bottom_front, edge_right_front,  edge_right_top,    edge_top_back,     edge_right_back,    edge_right_bottom};
	CASE(0b00101000) {edge_top_front,    edge_right_top,    edge_right_back,   edge_bottom_back,  edge_right_bottom,  edge_right_front };
	CASE(0b00100001) {edge_left_front,   edge_bottom_front, edge_right_bottom, edge_right_back,   edge_bottom_back,   edge_left_bottom };
	CASE(0b00010010) {edge_right_front,  edge_right_bottom, edge_bottom_back,  edge_left_back,    edge_left_bottom,   edge_bottom_front};
	CASE(0b10000100) {edge_left_front,   edge_left_top,     edge_top_back,     edge_right_back,   edge_right_top,     edge_top_front   };
	CASE(0b01001000) {edge_right_front,  edge_top_front,    edge_left_top,     edge_left_back,    edge_top_back,      edge_right_top   };

		// 2 Opposite Corners
	CASE(0b10000001) {edge_left_bottom,  edge_left_front,   edge_bottom_front, edge_right_back,   edge_right_top,    edge_top_back    };
	CASE(0b01000010) {edge_right_front,  edge_right_bottom, edge_bottom_front, edge_left_top,     edge_left_back,    edge_top_back    };
	CASE(0b00100100) {edge_left_front,   edge_left_top,     edge_top_front,    edge_right_bottom, edge_right_back,   edge_bottom_back };
	CASE(0b00011000) {edge_right_top,    edge_right_front,  edge_top_front,    edge_left_back,    edge_left_bottom,  edge_bottom_back };
		// 3 Corners
	CASE(0b01100001) {edge_left_bottom,  edge_left_front,   edge_bottom_front, edge_left_top,     edge_left_back,    edge_top_back,    edge_right_bottom, edge_right_back,   edge_bottom_back  };
	CASE(0b01001001) {edge_left_bottom,  edge_left_front,   edge_bottom_front, edge_left_top,     edge_left_back,    edge_top_back,    edge_right_top,    edge_right_front,  edge_top_front    };
	CASE(0b00101001) {edge_left_bottom,  edge_left_front,   edge_bottom_front, edge_right_bottom, edge_right_back,   edge_bottom_back, edge_right_top,    edge_right_front,  edge_top_front    };
	CASE(0b00010110) {edge_right_front,  edge_right_bottom, edge_bottom_front, edge_left_front,   edge_left_top,     edge_top_front,   edge_left_back,    edge_left_bottom,  edge_bottom_back  };
	CASE(0b10000110) {edge_right_front,  edge_right_bottom, edge_bottom_front, edge_left_front,   edge_left_top,     edge_top_front,   edge_right_back,   edge_right_top,    edge_top_back     };
	CASE(0b10010010) {edge_left_back,    edge_left_bottom,  edge_bottom_back,  edge_right_back,   edge_right_top,    edge_top_back,    edge_right_front,  edge_right_bottom, edge_bottom_front };
	CASE(0b10010100) {edge_left_back,    edge_left_bottom,  edge_bottom_back,  edge_right_back,   edge_right_top,    edge_top_back,    edge_left_front,   edge_left_top,     edge_top_front    };
	CASE(0b01101000) {edge_right_bottom, edge_right_back,   edge_bottom_back,  edge_left_top,     edge_left_back,    edge_top_back,    edge_right_top,    edge_right_front,  edge_top_front    };

	// 4 Corners
	CASE(0b01101001) {edge_right_bottom, edge_right_back,  edge_bottom_back,  edge_left_back,  edge_left_top,    edge_top_back, edge_right_top,    edge_right_front, edge_top_front, edge_left_bottom,  edge_left_front,  edge_bottom_front };

	//Flat
	CASE(0b00001111) {edge_left_top,     edge_right_top,    edge_right_bottom, edge_left_bottom };
	CASE(0b00110011) {edge_left_front,   edge_right_front,  edge_right_back,   edge_left_back   };
	CASE(0b01010101) {edge_bottom_back,  edge_top_back,     edge_top_front,    edge_bottom_front};

	// Edges
	CASE(0b00000011) {edge_right_front,  edge_right_bottom,  edge_left_bottom,  edge_left_front  };
	CASE(0b00001100) {edge_left_front,   edge_left_top,      edge_right_top,    edge_right_front };
	CASE(0b00110000) {edge_left_back,    edge_left_bottom,   edge_right_bottom, edge_right_back  };
	CASE(0b11000000) {edge_right_back,   edge_right_top,     edge_left_top,     edge_left_back   };
	CASE(0b00010001) {edge_left_front,   edge_bottom_front,  edge_bottom_back,  edge_left_back   };
	CASE(0b00100010) {edge_right_back,   edge_bottom_back,   edge_bottom_front, edge_right_front };
	CASE(0b01000100) {edge_left_back,    edge_top_back,      edge_top_front,    edge_left_front  };
	CASE(0b10001000) {edge_right_front,  edge_top_front,     edge_top_back,     edge_right_back  };
	CASE(0b00000101) {edge_left_top,     edge_top_front,     edge_bottom_front, edge_left_bottom };
	CASE(0b00001010) {edge_right_bottom, edge_bottom_front,  edge_top_front,    edge_right_top   };
	CASE(0b01010000) {edge_left_bottom,  edge_bottom_back,   edge_top_back,     edge_left_top    };
	CASE(0b10100000) {edge_right_top,    edge_top_back,      edge_bottom_back,  edge_right_bottom};

		// Hexagons
	CASE(0b00010111) {edge_right_front,  edge_right_bottom, edge_bottom_back,  edge_left_back,    edge_left_top,     edge_top_front   };
	CASE(0b00101011) {edge_right_top,    edge_right_back,   edge_bottom_back,  edge_left_bottom,  edge_left_front,   edge_top_front   };
	CASE(0b01110001) {edge_left_top,     edge_left_front,   edge_bottom_front, edge_right_bottom, edge_right_back,   edge_top_back    };
	CASE(0b01001101) {edge_left_bottom,  edge_left_back,    edge_top_back,     edge_right_top,    edge_right_front,  edge_bottom_front};

		// Zigzagoons
	CASE(0b00101110) {edge_left_top,     edge_right_top,    edge_right_back,   edge_bottom_back,  edge_bottom_front, edge_left_front  };
	CASE(0b01011100) {edge_top_back,     edge_right_top,    edge_right_front,  edge_left_front,   edge_left_bottom,  edge_bottom_back };
	CASE(0b11001010) {edge_top_front,    edge_left_top,     edge_left_back,    edge_right_back,   edge_right_bottom, edge_bottom_front};
	CASE(0b11000101) {edge_top_front,    edge_bottom_front, edge_left_bottom,  edge_left_back,    edge_right_back,   edge_right_top   };
	CASE(0b01110100) {edge_left_front,   edge_left_bottom,  edge_right_bottom, edge_right_back,   edge_top_back,     edge_top_front   };
	CASE(0b01000111) {edge_left_bottom,  edge_left_back,    edge_top_back,     edge_top_front,    edge_right_front,  edge_right_bottom};
	CASE(0b01110010) {edge_left_top,     edge_left_bottom,  edge_bottom_front, edge_right_front,  edge_right_back,   edge_top_back    };
	CASE(0b00100111) {edge_right_front,  edge_right_back,   edge_bottom_back,  edge_left_bottom,  edge_left_top,     edge_top_front   };
	CASE(0b10101100) {edge_left_top,     edge_top_back,     edge_bottom_back,  edge_right_bottom, edge_right_front,  edge_left_front  };
	CASE(0b11100010) {edge_left_top,     edge_left_back,    edge_bottom_back,  edge_bottom_front, edge_right_front,  edge_right_top   };
	CASE(0b00011011) {edge_right_top,    edge_right_bottom, edge_bottom_back,  edge_left_back,    edge_left_front,   edge_top_front   };
	CASE(0b10110001) {edge_left_back,    edge_left_front,   edge_bottom_front, edge_right_bottom, edge_right_top,    edge_top_back    };

		// Edges and corners
	CASE(0b00011100) {edge_left_back,    edge_bottom_back,  edge_left_bottom,  edge_left_front,   edge_right_front,  edge_right_top,    edge_left_top    };
	CASE(0b00110100) {edge_left_front,   edge_top_front,    edge_left_top,     edge_left_back,    edge_right_back,   edge_right_bottom, edge_left_bottom };
	CASE(0b00011001) {edge_right_front,  edge_right_top,    edge_top_front,    edge_left_front,   edge_left_back,    edge_bottom_back,  edge_bottom_front};
	CASE(0b10001001) {edge_left_front,   edge_left_bottom,  edge_bottom_front, edge_right_front,  edge_right_back,   edge_top_back,     edge_top_front   };
	CASE(0b00100110) {edge_top_front,    edge_left_top,     edge_left_front,   edge_bottom_front, edge_bottom_back,  edge_right_back,   edge_right_front };
	CASE(0b01000110) {edge_bottom_front, edge_right_bottom, edge_right_front,  edge_top_front,    edge_top_back,     edge_left_back,    edge_left_front  };
	CASE(0b01100010) {edge_left_back,    edge_left_top,     edge_top_back,     edge_right_back,   edge_right_front,  edge_bottom_front, edge_bottom_back };
	CASE(0b01100100) {edge_right_back,   edge_right_bottom, edge_bottom_back,  edge_left_back,    edge_left_front,   edge_top_front,    edge_top_back    };
	CASE(0b10010001) {edge_top_back,     edge_right_top,    edge_right_back,   edge_bottom_back,  edge_bottom_front, edge_left_front,   edge_left_back   };
	CASE(0b00101100) {edge_right_bottom, edge_bottom_back,  edge_right_back,   edge_right_top,    edge_left_top,     edge_left_front,   edge_right_front };
	CASE(0b10000011) {edge_right_back,   edge_top_back,     edge_right_top,    edge_left_bottom,  edge_right_bottom, edge_right_front,  edge_left_front  };
	CASE(0b00111000) {edge_right_top,    edge_top_front,    edge_right_front,  edge_right_bottom, edge_left_bottom,  edge_left_back,    edge_right_back  };
	CASE(0b11000001) {edge_left_bottom,  edge_bottom_front, edge_left_front,   edge_left_top,     edge_right_top,    edge_right_back,   edge_left_back   };
	CASE(0b01011000) {edge_right_top,    edge_top_front,    edge_right_front,  edge_top_back,     edge_bottom_back,  edge_left_bottom,  edge_left_top    };
	CASE(0b10100001) {edge_left_bottom,  edge_bottom_front, edge_left_front,   edge_bottom_back,  edge_top_back,     edge_right_top,    edge_right_bottom};
	CASE(0b10000101) {edge_right_back,   edge_top_back,     edge_right_top,    edge_bottom_front, edge_top_front,    edge_left_top,     edge_left_bottom };
	CASE(0b00100101) {edge_right_bottom, edge_bottom_back,  edge_right_back,   edge_bottom_front, edge_top_front,    edge_left_top,     edge_left_bottom };
	CASE(0b10011000) {edge_bottom_back,  edge_left_bottom,  edge_left_back,    edge_top_back,     edge_top_front,    edge_right_front,  edge_right_back  };
	CASE(0b10100100) {edge_left_front,   edge_top_front,    edge_left_top,     edge_bottom_back,  edge_top_back,     edge_right_top,    edge_right_bottom};
	CASE(0b00011010) {edge_left_back,    edge_bottom_back,  edge_left_bottom,  edge_top_front,    edge_bottom_front, edge_right_bottom, edge_right_top   };
	CASE(0b01010010) {edge_right_front,  edge_bottom_front, edge_right_bottom, edge_top_back,     edge_bottom_back,  edge_left_bottom,  edge_left_top    };
	CASE(0b01001010) {edge_left_top,     edge_top_back,     edge_left_back,    edge_top_front,    edge_bottom_front, edge_right_bottom, edge_right_top   };
	CASE(0b01000011) {edge_left_top,     edge_top_back,     edge_left_back,    edge_left_bottom,  edge_right_bottom, edge_right_front,  edge_left_front  };
	CASE(0b11000010) {edge_right_front,  edge_bottom_front, edge_right_bottom, edge_left_top,     edge_right_top,    edge_right_back,   edge_left_back   };

		// Double edges
	CASE(0b10100101) {edge_top_front,    edge_bottom_front, edge_left_bottom,  edge_left_top , edge_top_back,     edge_bottom_back,  edge_right_bottom, edge_right_top   };
	CASE(0b11000011) {edge_left_top,     edge_left_front,   edge_right_front,  edge_right_top, edge_left_bottom,  edge_left_back,    edge_right_back,   edge_right_bottom};
	CASE(0b10011001) {edge_bottom_back,  edge_bottom_front, edge_left_front,   edge_left_back, edge_top_back,     edge_top_front,    edge_right_front,  edge_right_back  };
		// Diagonal edges
	CASE(0b00010101) {edge_left_back,    edge_left_top,     edge_top_front,    edge_bottom_front, edge_bottom_back };
	CASE(0b01010001) {edge_left_top,     edge_left_front,   edge_bottom_front, edge_bottom_back,  edge_top_back    };
	CASE(0b01000101) {edge_left_bottom,  edge_left_back,    edge_top_back,     edge_top_front,    edge_bottom_front};
	CASE(0b00000111) {edge_top_front,    edge_right_front,  edge_right_bottom, edge_left_bottom,  edge_left_top    };
	CASE(0b00001101) {edge_right_front,  edge_bottom_front, edge_left_bottom,  edge_left_top,     edge_right_top   };
	CASE(0b00001011) {edge_right_bottom, edge_left_bottom,  edge_left_front,   edge_top_front,    edge_right_top   };
	CASE(0b00010011) {edge_right_bottom, edge_bottom_back,  edge_left_back,    edge_left_front,   edge_right_front };
	CASE(0b00100011) {edge_bottom_back,  edge_left_bottom,  edge_left_front,   edge_right_front,  edge_right_back  };
	CASE(0b00110001) {edge_bottom_front, edge_right_bottom, edge_right_back,   edge_left_back,    edge_left_front  };
	CASE(0b10101011) {edge_top_front,    edge_top_back,     edge_bottom_back,  edge_left_bottom,  edge_left_front  };
	CASE(0b11010101) {edge_bottom_back,  edge_right_back,   edge_right_top,    edge_top_front,    edge_bottom_front};
	CASE(0b01011101) {edge_top_back,     edge_right_top,    edge_right_front,  edge_bottom_front, edge_bottom_back };
	CASE(0b01010111) {edge_top_front,    edge_right_front,  edge_right_bottom, edge_bottom_back,  edge_top_back    };
	CASE(0b01110101) {edge_bottom_front, edge_right_bottom, edge_right_back,   edge_top_back,     edge_top_front   };
	CASE(0b11110001) {edge_right_bottom, edge_right_top,    edge_left_top,     edge_left_front,   edge_bottom_front};
	CASE(0b10001111) {edge_left_top,     edge_top_back,     edge_right_back,   edge_right_bottom, edge_left_bottom };
	CASE(0b00101111) {edge_right_top,    edge_right_back,   edge_bottom_back,  edge_left_bottom,  edge_left_top    };
	CASE(0b00011111) {edge_right_bottom, edge_bottom_back,  edge_left_back,    edge_left_top,     edge_right_top   };
	CASE(0b01001111) {edge_right_top,    edge_right_bottom, edge_left_bottom,  edge_left_back,    edge_top_back    };
	CASE(0b10110011) {edge_right_front,  edge_right_top,    edge_top_back,     edge_left_back,    edge_left_front  };
	CASE(0b01110011) {edge_right_back,   edge_top_back,     edge_left_top,     edge_left_front,   edge_right_front };
	CASE(0b00110111) {edge_left_back,    edge_left_top,     edge_top_front,    edge_right_front,  edge_right_back  };
	CASE(0b00111011) {edge_left_front,   edge_top_front,    edge_right_top,    edge_right_back,   edge_left_back   };
	CASE(0b11001101) {edge_left_back,    edge_right_back,   edge_right_front,  edge_bottom_front, edge_left_bottom };
		// Diagonal edges and corners
	CASE(0b10010101) {edge_left_back,    edge_left_top,     edge_top_front,    edge_bottom_front, edge_bottom_back , edge_right_top,   edge_right_back,   edge_top_back     };
	CASE(0b01011001) {edge_left_top,     edge_left_front,   edge_bottom_front, edge_bottom_back,  edge_top_back    , edge_right_front, edge_right_top,    edge_top_front    };
	CASE(0b01010110) {edge_left_front,   edge_left_bottom,  edge_bottom_back,  edge_top_back,     edge_top_front   , edge_right_front, edge_right_bottom, edge_bottom_front };
	CASE(0b01100101) {edge_left_bottom,  edge_left_back,    edge_top_back,     edge_top_front,    edge_bottom_front, edge_right_back,  edge_right_bottom, edge_bottom_back  };
	CASE(0b10000111) {edge_top_front,    edge_right_front,  edge_right_bottom, edge_left_bottom,  edge_left_top    , edge_right_top,   edge_right_back,   edge_top_back     };
	CASE(0b00101101) {edge_right_front,  edge_bottom_front, edge_left_bottom,  edge_left_top,     edge_right_top   , edge_right_back,  edge_right_bottom, edge_bottom_back  };
	CASE(0b00011110) {edge_bottom_front, edge_left_front,   edge_left_top,     edge_right_top,    edge_right_bottom, edge_left_back,   edge_left_bottom,  edge_bottom_back  };
	CASE(0b01001011) {edge_top_front,    edge_left_front,   edge_left_bottom,  edge_right_bottom, edge_right_top   , edge_left_top,    edge_left_back,    edge_top_back     };
	CASE(0b01101100) {edge_right_top,    edge_top_back,     edge_left_back,    edge_left_front,   edge_right_front , edge_right_back,  edge_right_bottom, edge_bottom_back  };
	CASE(0b10011100) {edge_top_back,     edge_left_top,     edge_left_front,   edge_right_front,  edge_right_back  , edge_left_back,   edge_left_bottom,  edge_bottom_back  };
	CASE(0b11001001) {edge_left_top,     edge_top_front,    edge_right_front,  edge_right_back,   edge_left_back   , edge_left_front,  edge_left_bottom,  edge_bottom_front };
	CASE(0b11000110) {edge_top_front,    edge_right_top,    edge_right_back,   edge_left_back,    edge_left_front  , edge_right_front, edge_right_bottom, edge_bottom_front };
}

struct Mesh {
	List<Vertex> vertices;
	List<u32> indices;
};

Mesh generate_mesh(Grid *grids[]) {
	// sometimes weird spikes are generated when neighbor is not available
	// this is caused by accessing uninitialized indices. for now i initialize indices with invalid values
	// and perform a check at :IndexCheck.
	// this is inefficient because of initialization of `index_grid` and extra check in triangle generation loop

	assert(grids[GridIndex_self]);

	u32 const extension_amount = 2;
	u32 const extended_resolution = grid_resolution + extension_amount;

	// This grid includes cells of neighboring chunks
	Array3<Cell, extended_resolution, extended_resolution, extended_resolution> full_grid = {};

	for (s32 z = 0; z < grid_resolution; ++z)
	for (s32 y = 0; y < grid_resolution; ++y)
	for (s32 x = 0; x < grid_resolution; ++x) {
		full_grid.at(x, y, z) = grids[GridIndex_self]->at(x, y, z);
	}
	if (grids[GridIndex_x]) {
		for (s32 z = 0; z < grid_resolution; ++z)
		for (s32 y = 0; y < grid_resolution; ++y)
		for (s32 x = 0; x < extension_amount; ++x) {
			full_grid.at(grid_resolution + x, y, z) = grids[GridIndex_x]->at(x, y, z);
		}
	}
	if (grids[GridIndex_y]) {
		for (s32 z = 0; z < grid_resolution; ++z)
		for (s32 y = 0; y < extension_amount; ++y)
		for (s32 x = 0; x < grid_resolution; ++x) {
			full_grid.at(x, grid_resolution + y, z) = grids[GridIndex_y]->at(x, y, z);
		}
	}
	if (grids[GridIndex_z]) {
		for (s32 z = 0; z < extension_amount; ++z)
		for (s32 y = 0; y < grid_resolution; ++y)
		for (s32 x = 0; x < grid_resolution; ++x) {
			full_grid.at(x, y, grid_resolution + z) = grids[GridIndex_z]->at(x, y, z);
		}
	}
	if (grids[GridIndex_xy]) {
		for (s32 z = 0; z < grid_resolution; ++z)
		for (s32 y = 0; y < extension_amount; ++y)
		for (s32 x = 0; x < extension_amount; ++x) {
			full_grid.at(grid_resolution + x, grid_resolution + y, z) = grids[GridIndex_xy]->at(x, y, z);
		}
	}
	if (grids[GridIndex_xz]) {
		for (s32 z = 0; z < extension_amount; ++z)
		for (s32 y = 0; y < grid_resolution; ++y)
		for (s32 x = 0; x < extension_amount; ++x) {
			full_grid.at(grid_resolution + x, y, grid_resolution + z) = grids[GridIndex_xz]->at(x, y, z);
		}
	}
	if (grids[GridIndex_yz]) {
		for (s32 z = 0; z < extension_amount; ++z)
		for (s32 y = 0; y < extension_amount; ++y)
		for (s32 x = 0; x < grid_resolution; ++x) {
			full_grid.at(x, grid_resolution + y, grid_resolution + z) = grids[GridIndex_yz]->at(x, y, z);
		}
	}
	if (grids[GridIndex_xyz]) {
		for (s32 z = 0; z < extension_amount; ++z)
		for (s32 y = 0; y < extension_amount; ++y)
		for (s32 x = 0; x < extension_amount; ++x) {
			full_grid.at(grid_resolution + x, grid_resolution + y, grid_resolution + z) = grids[GridIndex_xyz]->at(x, y, z);
		}
	}

	Array3<Vertex, extended_resolution - 1, extended_resolution - 1, extended_resolution - 1> vertex_grid = {};
	for (auto &v : vertex_grid) {
		v.position = {0, 1000, 0};
	}

	Array3<bool, extended_resolution - 1, extended_resolution - 1, extended_resolution - 1> vert_init_grid = {};

	auto add_point = [&](s8 samples[8], u32 x, u32 y, u32 z) {
		for (u32 i = 0; i < 8; ++i) {
			if (samples[i] == distance_cell_surface) {
				// Invalid voxel means that neighbor's voxels are not available
				vertex_grid.at(x, y, z).position.x = -1337;
				return;
			}
		}

		u8 intersection_index = 0;
		for (u8 i = 0; i < 8; ++i) {
			intersection_index |= (samples[i] > distance_cell_surface) << i;
		}

		if (intersection_index == 0 || intersection_index == 255) {
			vertex_grid.at(x, y, z).position.x = -1337;
			return;
		}

		auto const &edges = intersected_edges[intersection_index];

		//x = b + (a - b) * t;
		//x - b = (a - b) * t;
		//t = (x - b) / (a - b)


		Vertex vertex = {};
		for (u32 i = 0; i < edges.size; ++i) {
			auto edge = edges[i];
			f32 t = (f32)samples[edge.vertex_a] / ((s16)samples[edge.vertex_a] - samples[edge.vertex_b]);
			assert(0 <= t && t <= 1);
			v3f point = lerp(
				base_positions[edge.vertex_a],
				base_positions[edge.vertex_b],
				V3f(t)
			);
			vertex.position += point;
		}
		vertex.position /= (f32)edges.size;
		vertex.position += V3f(x, y, z);

		assert(x <= vertex.position.x && vertex.position.x <= x + 1);
		assert(y <= vertex.position.y && vertex.position.y <= y + 1);
		assert(z <= vertex.position.z && vertex.position.z <= z + 1);

		v3f normal;
		normal.x =
			samples[right_bottom_front] - samples[left_bottom_front] +
			samples[right_bottom_back ] - samples[left_bottom_back ] +
			samples[right_top_front   ] - samples[left_top_front   ] +
			samples[right_top_back    ] - samples[left_top_back    ];
		normal.y =
			samples[left_top_front ] - samples[left_bottom_front ] +
			samples[left_top_back  ] - samples[left_bottom_back  ] +
			samples[right_top_front] - samples[right_bottom_front] +
			samples[right_top_back ] - samples[right_bottom_back ];
		normal.z =
			samples[left_bottom_front ] - samples[left_bottom_back ] +
			samples[left_top_front    ] - samples[left_top_back    ] +
			samples[right_bottom_front] - samples[right_bottom_back] +
			samples[right_top_front   ] - samples[right_top_back   ];

		vertex.normal = normalize(normal);
#if DEBUG_VERTEX
		vertex.debug_distance = full_grid.at(x,y,z).debug_distance;
#endif
		vertex_grid.at(x, y, z) = vertex;
		vert_init_grid.at(x, y, z) = true;
	};

	for (u32 z = 0; z < extended_resolution - 1; ++z) {
	for (u32 y = 0; y < extended_resolution - 1; ++y) {
	for (u32 x = 0; x < extended_resolution - 1; ++x) {
		s8 samples[8];
		samples[  left_bottom_back] = full_grid.at(x + 0, y + 0, z + 0).distance;
		samples[ right_bottom_back] = full_grid.at(x + 1, y + 0, z + 0).distance;
		samples[     left_top_back] = full_grid.at(x + 0, y + 1, z + 0).distance;
		samples[    right_top_back] = full_grid.at(x + 1, y + 1, z + 0).distance;
		samples[ left_bottom_front] = full_grid.at(x + 0, y + 0, z + 1).distance;
		samples[right_bottom_front] = full_grid.at(x + 1, y + 0, z + 1).distance;
		samples[    left_top_front] = full_grid.at(x + 0, y + 1, z + 1).distance;
		samples[   right_top_front] = full_grid.at(x + 1, y + 1, z + 1).distance;

		add_point(samples, x, y, z);
	}
	}
	}

	Array3<u32, extended_resolution - 1, extended_resolution - 1, extended_resolution - 1> index_grid;
	for (auto &index : index_grid) {
		index = -1;
	}

	List<Vertex> vertices;

	for (s32 z = 0; z < extended_resolution - 1; ++z) {
	for (s32 y = 0; y < extended_resolution - 1; ++y) {
	for (s32 x = 0; x < extended_resolution - 1; ++x) {
		auto &P1 = vertex_grid.at(x, y, z);
		if (P1.position.x == -1337)
			continue;

		index_grid.at(x, y, z) = vertices.size;
		vertices.add(P1);
		assert(vert_init_grid.at(x, y, z));
	}
	}
	}

	List<u32> indices;

	// Start at 1 because chunk behind will have necessary faces
	for (s32 z = 1; z < extended_resolution - 1; ++z) {
	for (s32 y = 1; y < extended_resolution - 1; ++y) {
	for (s32 x = 1; x < extended_resolution - 1; ++x) {
		v3s ps = {x, y, z};

		s8 samples[8];
		samples[left_bottom_back ] = full_grid.at(x + 0, y + 0, z + 0).distance;
		samples[right_bottom_back] = full_grid.at(x + 1, y + 0, z + 0).distance;
		samples[left_top_back    ] = full_grid.at(x + 0, y + 1, z + 0).distance;
		samples[left_bottom_front] = full_grid.at(x + 0, y + 0, z + 1).distance;

		if (samples[left_bottom_back ] == distance_cell_surface
		 || samples[right_bottom_back] == distance_cell_surface
		 || samples[left_top_back    ] == distance_cell_surface
		 || samples[left_bottom_front] == distance_cell_surface)
			continue;

		static constexpr Edge axes[] = {
			edge_bottom_back, // x
			edge_left_back,   // y
			// Swap vertices here so face orientation can be determined with one comparison
			{ edge_left_bottom.vertex_b, edge_left_bottom.vertex_a }, // z
		};

		for (u32 edge_index = 0; edge_index < 3; ++edge_index) {
			auto edge = axes[edge_index];
			if ((samples[edge.vertex_a] > distance_cell_surface) != (samples[edge.vertex_b] > distance_cell_surface)) {
				// (x >= grid_resolution) is here because we don't need faces in that
				// direction at the last layer as they are present in the next chunk
				static constexpr v3s vertex_offsets[3][4] = {
					{
						{ 0,-1, 0},
						{ 0,-1,-1},
						{ 0,-1,-1},
						{ 0, 0,-1},
					},
					{
						{-1, 0, 0},
						{-1, 0,-1},
						{-1, 0,-1},
						{ 0, 0,-1},
					},
					{
						{-1, 0, 0},
						{-1,-1, 0},
						{-1,-1, 0},
						{ 0,-1, 0},
					},
				};

				auto get_index = [&](v3s at) {
					return index_grid.at(at.x, at.y, at.z);
				};

				u32 start_vertex = vertices.size;

				u32 temp_indices[6];

				temp_indices[0] = get_index(ps);
				temp_indices[1] = get_index(ps + vertex_offsets[edge_index][0]);
				temp_indices[2] = get_index(ps + vertex_offsets[edge_index][1]);
				temp_indices[3] = temp_indices[0];
				temp_indices[4] = get_index(ps + vertex_offsets[edge_index][2]);
				temp_indices[5] = get_index(ps + vertex_offsets[edge_index][3]);

				// :IndexCheck
				for (auto &index : temp_indices) {
					if (index == -1) {
						// TODO: there should be a way to not go through bad indices
						goto next_cell;
					}
				}

				if (samples[edge.vertex_b] > distance_cell_surface) {
					std::swap(temp_indices[0], temp_indices[1]);
					std::swap(temp_indices[3], temp_indices[4]);
				}

				indices += Span(temp_indices);

			}
		}
	next_cell:;
	}
	}
	}

	//calculate_normals(as_span(vertices), as_span(indices));

	return {vertices, indices};
}

}

void generate_distance_grid(Chunk &chunk) {
	for_v3 (v3s, local, {}, <, V3s(grid_resolution), ++) {
		v3s global = chunk.position * grid_resolution + local;
		f32 d =
			  value_noise_v3s_smooth(global, 32) * 32
			+ value_noise_v3s_smooth(global, 16) * 16
			+ value_noise_v3s_smooth(global,  8) *  8
			+ value_noise_v3s_smooth(global,  4) *  4
			+ global.y;
		//f32 d = (s32)grid_resolution/2 - (s32)global.y;
		s32 v = map_clamped<f32>(d, -1, 1, distance_cell_min, distance_cell_max);
		if (v == distance_cell_surface)
			++v;
		chunk.grid.at(local).distance = v;
	}
}

void generate_buffers(Chunk &chunk, MeshGenerator::Mesh const &mesh) {
	auto &vertices = mesh.vertices;
	auto &indices = mesh.indices;
	chunk.index_count = indices.size;

	free(chunk.vertices);
	free(chunk.indices);
	chunk.vertices = vertices;
	chunk.indices = indices;

	if (chunk.vertex_array) {
		glBindVertexArray(chunk.vertex_array);
		glBindBuffer(GL_ARRAY_BUFFER, chunk.vertex_buffer);
	} else {
		glGenVertexArrays(1, &chunk.vertex_array);
		glGenBuffers(count_of(chunk.buffers), chunk.buffers);

		glBindVertexArray(chunk.vertex_array);
		glBindBuffer(GL_ARRAY_BUFFER, chunk.vertex_buffer);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(Vertex), (void *)offsetof(Vertex, position));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(Vertex), (void *)offsetof(Vertex, normal));
#if DEBUG_VERTEX
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 1, GL_FLOAT, false, sizeof(Vertex), (void *)offsetof(Vertex, debug_distance));
#endif
	}

	glBufferData(GL_ARRAY_BUFFER, vertices.size * sizeof(vertices[0]), vertices.data, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, chunk.index_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size * sizeof(indices[0]), indices.data, GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

umm get_hash(v3s const &v) {
	return (umm)v.z * 256 + (umm)v.y * 16 + (umm)v.x;
}

StaticHashMap<v3s, Chunk, 256> chunks;

struct RemeshRequest {
	Chunk *target;
	Grid *grids[GridIndex_count];
	bool operator==(RemeshRequest const &that) {
		return target == that.target;
	}
};

umm get_hash(RemeshRequest const &request) {
	return (umm)request.target / 8;
}

LinearSet<RemeshRequest> remesh_requests;
LinearSet<Chunk *> generation_requests;

Grid *try_get_grid(Chunk *chunk) {
	return chunk ? &chunk->grid : 0;
}

bool add_remesh_request_for(Chunk *chunk, bool force) {
	RemeshRequest request;
	request.target = chunk;
	request.grids[GridIndex_self] = &chunk->grid;
	request.grids[GridIndex_x  ] = try_get_grid(chunks.find(chunk->position + v3s{1,0,0}));
	request.grids[GridIndex_y  ] = try_get_grid(chunks.find(chunk->position + v3s{0,1,0}));
	request.grids[GridIndex_z  ] = try_get_grid(chunks.find(chunk->position + v3s{0,0,1}));
	request.grids[GridIndex_xy ] = try_get_grid(chunks.find(chunk->position + v3s{1,1,0}));
	request.grids[GridIndex_xz ] = try_get_grid(chunks.find(chunk->position + v3s{1,0,1}));
	request.grids[GridIndex_yz ] = try_get_grid(chunks.find(chunk->position + v3s{0,1,1}));
	request.grids[GridIndex_xyz] = try_get_grid(chunks.find(chunk->position + v3s{1,1,1}));
	u8 neighbor_count = 0;
	for (u32 i = 0; i < 7; ++i) {
		neighbor_count += (request.grids[i + 1] != 0);
	}
	if ((neighbor_count == chunk->neighbor_count) && chunk->vertex_array && !force) {
		return false;
	}
	assert(neighbor_count >= chunk->neighbor_count);
	chunk->neighbor_count = neighbor_count;
	remesh_requests.insert(request);
	return true;
}

void load_nearby_chunks(v3s camera_chunk_position) {
	s32 const load_distance = 1;
	for (s32 z = -load_distance; z <= +load_distance; ++z)
	for (s32 y = -load_distance; y <= +load_distance; ++y)
	for (s32 x = -load_distance; x <= +load_distance; ++x) {
		v3s pos = camera_chunk_position + v3s{x,y,z};
		Chunk *chunk = chunks.find(pos);

		if (chunk) {
			assert(chunk->position == pos);
		} else {
			chunk = &chunks.get_or_insert(pos);
			chunk->position = pos;
			generation_requests.insert(chunk);
		}

		//try_add_remesh_request_for(chunk);
	}
	for (s32 z = -load_distance; z <= +load_distance; ++z)
	for (s32 y = -load_distance; y <= +load_distance; ++y)
	for (s32 x = -load_distance; x <= +load_distance; ++x) {
		v3s pos = camera_chunk_position + v3s{x,y,z};
		if (all_true(v3s{x,y,z} == v3s{}) && key_down(Key_f1)) {
			debug_break();
		}
		Chunk *chunk = chunks.find(pos);
		assert(chunk);
		assert(chunk->position == pos);
		add_remesh_request_for(chunk, false);
	}

	for (auto chunk : generation_requests) {
		generate_distance_grid(*chunk);
	}
	generation_requests.clear();

	for (auto request : remesh_requests) {
		generate_buffers(*request.target, MeshGenerator::generate_mesh(request.grids));
	}
	remesh_requests.clear();
}

struct Position {
	v3s chunk;
	v3f local;
};

void append(StringBuilder &b, Position p) {
	return append(b, (v3f)(p.chunk * grid_resolution) + p.local);
}

struct RaycastHit {
	bool hit;
	Chunk *chunk;
	Position position;
	f32 distance;

	operator bool() {
		return hit;
	}
};

Cell find_cell(v3s global_cell_pos) {
	v3s chunk_position = floor(global_cell_pos, grid_resolution);
	Chunk *chunk = chunks.find(chunk_position);
	if (chunk) {
		return chunk->grid.at(global_cell_pos - chunk_position);
	}
}

Array<Cell, 8> find_cells(v3s global_start_pos) {
	Array<Cell, 8> result;
	v3s cell_positions[] {
		global_start_pos + v3s{0,0,0},
		global_start_pos + v3s{0,0,1},
		global_start_pos + v3s{0,1,0},
		global_start_pos + v3s{0,1,1},
		global_start_pos + v3s{1,0,0},
		global_start_pos + v3s{1,0,1},
		global_start_pos + v3s{1,1,0},
		global_start_pos + v3s{1,1,1},
	};

	for (u32 i = 0; i < 8; ++i) {
		result[i] = find_cell(cell_positions[i]);
	}

	return result;
}

// `origin` is in `chunk`'s local space
::RaycastHit raycast(Chunk *chunk, v3f origin, v3f direction) {
	::RaycastHit hit = {};
	hit.distance = INFINITY;

	// Distance field cell check count with resolution of 32:
	//    Worst case        - 96

	// Triangle test count with resolution of 32:
	//     Worst case       - 65k
	//     For flat surface - 2k

#if 1
	// Intersect with distance field

	//if (!intersects(aabb_min_max(v3s{}, V3s(grid_resolution)), line_begin_end(origin, origin + direction * )))

	v3f cellf = origin;
	v3s cell = floor_to_int(cellf);

	v3f direction_sign = sign(direction);
	v3f direction_positive = {
		is_positive(direction.x),
		is_positive(direction.y),
		is_positive(direction.z),
	};

	auto calculate_next_cell = [&] {
		f32 d = INFINITY;
		v3f p0 = lerp(ceil(cellf) - 1, floor(cellf) + 1, direction_positive);

		static constexpr v3f normals[] {
			{1,0,0},
			{0,1,0},
			{0,0,1},
		};
		static constexpr v3s deltas_for_next_cell[] {
			{1,0,0},
			{0,1,0},
			{0,0,1},
		};

		v3s delta_for_next_cell;

		for (u32 plane_index = 0; plane_index < 3; ++plane_index) {
			v3f n = normals[plane_index] * direction_sign.s[plane_index];

			// if `denom` is 0, line is parallel to plane
			f32 denom = dot(direction, n);
			if (denom) {
				f32 new_d = dot(p0 - cellf, n) / denom;
				if (new_d < d) {
					d = new_d;
					delta_for_next_cell = deltas_for_next_cell[plane_index] * direction_sign.s[plane_index];
				}
			}
		}

		cellf = cellf + direction * d;
	};

	while (!in_bounds(cell, aabb_min_max(v3s{}, V3s(grid_resolution)))) {
		calculate_next_cell();
	}

#else
	// Intersect with triangles
	for (u32 i = 0; i < chunk->indices.size; i += 3) {
		v3f a = chunk->vertices[chunk->indices[i + 0]].position;
		v3f b = chunk->vertices[chunk->indices[i + 1]].position;
		v3f c = chunk->vertices[chunk->indices[i + 2]].position;

		if (auto new_hit = raycast(ray_begin_dir(origin, direction), triangle<v3f>{a, b, c})) {
			hit.hit = true;
			f32 dist = distance(origin, new_hit.position);
			if (dist < hit.distance) {
				hit.distance = dist;
				hit.position.local = new_hit.position;
			}
		}
	}
#endif
	hit.position.chunk = chunk->position;
	hit.chunk = chunk;
	return hit;
}
::RaycastHit raycast(v3f origin, v3f direction, f32 max_distance) {
	::RaycastHit hit = {};
	hit.distance = INFINITY;

	v3f minf = origin;
	v3f maxf = origin + direction * max_distance;
	minmax(minf, maxf, minf, maxf);
	minf -= V3f(2); // mesh can exceed chunk bounds by 2 units on positive directions

	v3s mins = floor_to_int(minf / grid_resolution);
	v3s maxs =  ceil_to_int(maxf / grid_resolution);

	for (s32 z = mins.z; z <= maxs.z; ++z)
	for (s32 y = mins.y; y <= maxs.y; ++y)
	for (s32 x = mins.x; x <= maxs.x; ++x) {
		v3s chunk_pos = {x,y,z};
		Chunk *chunk = chunks.find(chunk_pos);
		if (chunk) {
			if (auto new_hit = raycast(chunk, origin - (v3f)(chunk_pos*grid_resolution), direction)) {
				if (new_hit.distance <= max_distance) {
					if (new_hit.distance < hit.distance) {
						hit = new_hit;
					}
				}
			}
		}
	}

	return hit;
}

s32 tl_main(Span<Span<utf8>> arguments) {
	init_printer();
	defer { deinit_printer(); };

	current_printer = console_printer;

	MeshGenerator::init();

	sizeof(chunks);

	show_console_window();
	current_printer = console_printer;

	CreateWindowInfo info;
	info.on_draw = [](Window &window) {
		glViewport(window.client_size);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		m4 projection_matrix = m4::perspective_right_handed((f32)window.client_size.x / window.client_size.y, radians(90), 0.01f, 256.f);

		camera_rotation.y += window.mouse_delta.x * 0.003f;
		camera_rotation.x -= window.mouse_delta.y * 0.003f;
		m4 camera_rotation_matrix = m4::rotation_zxy(camera_rotation);

		v3f camera_delta = {};
		if (key_held('A')) --camera_delta.x;
		if (key_held('D')) ++camera_delta.x;
		if (key_held('W')) --camera_delta.z;
		if (key_held('S')) ++camera_delta.z;
		camera_position += m4::rotation_zxy(0, camera_rotation.y, 0) * normalize(camera_delta, {}) * frame_time * 16;

		f32 const camera_height = 2;

		if (auto hit = raycast(camera_position, v3f{0,-1,0}, camera_height)) {
			camera_position = (v3f)(hit.position.chunk*grid_resolution) + hit.position.local + v3f{0, camera_height, 0};
		} else {
			camera_position.y -= frame_time * 10;
			// Second raycast here once again to ensure that camera did not got through the surface
			if (auto hit = raycast(camera_position, v3f{0,-1,0}, camera_height)) {
				camera_position = (v3f)(hit.position.chunk*grid_resolution) + hit.position.local + v3f{0, camera_height, 0};
			}
		}
		if (key_down('R')) {
			camera_position = {};
		}

		v3s camera_chunk_position = floor_to_int(camera_position / grid_resolution);

		static bool do_load = true;
		if (key_down(Key_f2)) {
			do_load = !do_load;
		}

		if (do_load) {
			load_nearby_chunks(camera_chunk_position);
		}
		if (key_held(' ')) {

			f32 const brush_radius = 8;

			if (auto hit = raycast(camera_position,  (camera_rotation_matrix * v4f{0,0,-1,0}).xyz, 64)) {
				v3s min_affected_chunk = floor_to_int((hit.position.local - brush_radius) / grid_resolution) + hit.chunk->position;
				v3s max_affected_chunk =  ceil_to_int((hit.position.local + brush_radius) / grid_resolution) + hit.chunk->position;

				v3s min_cell_global = floor_to_int(hit.position.local - brush_radius) + hit.chunk->position * grid_resolution;
				v3s max_cell_global =  ceil_to_int(hit.position.local + brush_radius) + hit.chunk->position * grid_resolution;


				for (s32 affected_chunk_z = min_affected_chunk.z; affected_chunk_z <= max_affected_chunk.z; ++affected_chunk_z)
				for (s32 affected_chunk_y = min_affected_chunk.y; affected_chunk_y <= max_affected_chunk.y; ++affected_chunk_y)
				for (s32 affected_chunk_x = min_affected_chunk.x; affected_chunk_x <= max_affected_chunk.x; ++affected_chunk_x) {
					v3s affected_chunk_pos = {affected_chunk_x, affected_chunk_y, affected_chunk_z};
					Chunk *affected_chunk = chunks.find(affected_chunk_pos);
					if (affected_chunk) {
						v3s min_cell_local = max(min_cell_global - affected_chunk->position * grid_resolution, V3s(0));
						v3s max_cell_local = min(max_cell_global - affected_chunk->position * grid_resolution, V3s(grid_resolution - 1));

						v3f hit_position_local = hit.position.local + (v3f)((hit.chunk->position - affected_chunk->position) * grid_resolution);

#if DEBUG_VERTEX
						for (auto &cell : affected_chunk->grid) {
							cell.debug_distance = 0;
						}
#endif

						for (s32 z = min_cell_local.z; z <= max_cell_local.z; ++z)
						for (s32 y = min_cell_local.y; y <= max_cell_local.y; ++y)
						for (s32 x = min_cell_local.x; x <= max_cell_local.x; ++x) {
							v3s cell_index = {x,y,z};
							v3s cell_global = cell_index + affected_chunk->position * grid_resolution;

							f32 dist = distance((v3f)cell_index, hit_position_local);

#if DEBUG_VERTEX
							affected_chunk->grid.at(cell_index).debug_distance = dist;
#endif

							if (dist < brush_radius) {
								s32 cell = affected_chunk->grid.at(cell_index).distance;
								cell += brush_radius - dist;
								if (cell == distance_cell_surface) {
									cell += 1;
								}
								affected_chunk->grid.at(cell_index).distance = clamp(cell, distance_cell_min, distance_cell_max);
							}
						}
						add_remesh_request_for(affected_chunk, true);
					} else {
						print("TODO: deltas for unloaded chunks");
					}
				}

			}
		}

		m4 camera_matrix = projection_matrix * m4::rotation_yxz(-camera_rotation) * m4::translation(-camera_position);

		glEnable(GL_POLYGON_OFFSET_LINE);
		glPolygonOffset(0, -10);

		for (auto &[position, chunk] : chunks) {
			m4 mvp_matrix = camera_matrix * m4::translation((v3f)(chunk.position * grid_resolution));

			glBindVertexArray(chunk.vertex_array);

			glUseProgram(surface_shader);
			set_uniform(surface_shader, "mvp_matrix", mvp_matrix);
			glDrawElements(GL_TRIANGLES, chunk.index_count, GL_UNSIGNED_INT, 0);

			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			glUseProgram(wireframe_shader);
			set_uniform(wireframe_shader, "mvp_matrix", mvp_matrix);
			glDrawElements(GL_TRIANGLES, chunk.index_count, GL_UNSIGNED_INT, 0);

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glBindVertexArray(0);
		}

		present();

		frame_time = reset(frame_timer);
		time += frame_time;

		clear_temporary_storage();

		set_title(&window, tformat(u8"%", floor_to_int(camera_position / grid_resolution)));
	};
	if (!create_window(&window, info)) {
		print("Failed to create window\n");
		return 1;
	}

	init_opengl(window->handle, true);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	chunks = {};
	remesh_requests = {};
	generation_requests = {};

	auto vertex_shader = create_shader(GL_VERTEX_SHADER, 330, true, R"(
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
//layout(location=2) in float debug_distance;

out vec3 vertex_normal;
out vec3 vertex_color;

uniform mat4 mvp_matrix;
void main() {
	gl_Position = mvp_matrix * vec4(position, 1);
	vertex_normal = normal;
	vertex_color = position / 32;
	//vertex_color = vec3(debug_distance / 16);
}
)"s);
	auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, 330, true, R"(
in vec3 vertex_normal;
in vec3 vertex_color;
out vec4 fragment_color;
void main() {
	float diffuse = dot(vertex_normal,normalize(vec3(1,3,2)));

	fragment_color = vec4(vertex_color * diffuse, 1);
	fragment_color = vec4(vertex_color, 1);
}
)"s);
	surface_shader = create_program(vertex_shader, fragment_shader);

	vertex_shader = create_shader(GL_VERTEX_SHADER, 330, true, R"(
layout(location=0) in vec3 position;
uniform mat4 mvp_matrix;
void main() {
	gl_Position = mvp_matrix * vec4(position, 1);
}
)"s);
	fragment_shader = create_shader(GL_FRAGMENT_SHADER, 330, true, R"(
out vec4 fragment_color;
void main() {
	fragment_color = vec4(0.25, 0.25, 0.5, 0);
}
)"s);
	wireframe_shader = create_program(vertex_shader, fragment_shader);

	frame_timer = create_precise_timer();
	while (update(window)) {
	}

	return 0;
}
