#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdalign.h>
#include <immintrin.h>

// TODO: rip through the array and count bucket sizes
// simultaneously make a uint8_t [10,000] mapping
#define DEBUG

#ifdef DEBUG
#include <sys/resource.h>
#define SYS_EVAL()                                                             \
	do {                                                                   \
		struct rlimit lim;                                             \
		getrlimit(RLIMIT_STACK, &lim);                                 \
		printf("stack limit: %lu\n", lim.rlim_cur / 1024);             \
		if (__builtin_cpu_supports("avx2")) {                          \
			printf("AVX2 supported\n");                            \
		} else {                                                       \
			printf("AVX2 unsupported\n");                          \
		}                                                              \
		if (__builtin_cpu_supports("avx512f")) {                       \
			printf("AVX512 supported\n");                          \
		}                                                              \
	} while (0)

#include <time.h>
#define TIME(code)                                                             \
	do {                                                                   \
		struct timespec start, end;                                    \
		clock_gettime(CLOCK_MONOTONIC, &start);                        \
		code clock_gettime(CLOCK_MONOTONIC, &end);                     \
		long long ns = (long long)((end.tv_sec - start.tv_sec) *       \
					   1000000000LL) +                     \
			       (end.tv_nsec - start.tv_nsec);                  \
		printf("%s\ntook %lld ns at line %d", #code, ns, __LINE__);    \
	} while (0)

#define CHECK_ALIGN(ptr, bytes)                                                \
	do {                                                                   \
		if (((long)ptr % bytes) == 0) {                                \
			printf("%s aligned to %d\n", #ptr, bytes);             \
		} else {                                                       \
			printf("%s not aligned to %d\n", #ptr, bytes);         \
		}                                                              \
	} while (0)
#define LOG(msg, ...)                                                          \
	do {                                                                   \
		printf(msg, ##__VA_ARGS__);                                    \
	} while (0)
#else
#define SYS_EVAL()
#define TIME(code) code
#define CHECK_ALIGN(ptr, bytes)
#define LOG(msg)
#endif

#define OFFSET 1000000001
#define KEY_BITS 11U
#define NUM_BUCKETS 2048U // 1 << KEY_BITS

__attribute__((target("avx2"))) //
int* twoSum(int* nums, int n, int target, int* returnsize)
{

	// SYS_EVAL();
	// CHECK_ALIGN(nums, 32);
	int* retvals = malloc(sizeof(int) * 2);
	if (retvals == NULL) {
		*returnsize = 0;
		return NULL;
	}

	// Shift target into positive range
	target = target + OFFSET;
	LOG("Target post shift: %032b\n", target);

	// Find most significant bit of target
	uint8_t lz = __builtin_clz(target);
	uint8_t msb_pos = 32 - lz;
	// shift = position of most significant bits of target or bottom bits
	uint8_t key_shift =
		(msb_pos > (int)KEY_BITS) ? msb_pos - (int)KEY_BITS : 0;
	// Shift 1 left by KEYBITS then subtract 1 to set all bits below the 1
	uint32_t key_mask = (1U << KEY_BITS) - 1U;
	// shift the key mask to use only the used bits
	key_mask <<= key_shift;
	LOG("Keymask: %032b\n", key_mask);

	uint8_t bkt_accumulators[8 * NUM_BUCKETS] = {0};
	const __m256i OFFSET_V = _mm256_set1_epi32(OFFSET);
	const __m256i KEY_MASK_V = _mm256_set1_epi32((int)key_mask);

	// TODO: Pessimistically repeat this loop for subsets of n such that
	// uint8 overflow is impossible
	for (int i = 0; i <= n - 16; i += 16) {
		// Load 8 nums
		__m256i v_nums1 = _mm256_loadu_si256((__m256i*)(nums + i));
		__m256i v_nums2 = _mm256_loadu_si256((__m256i*)(nums + i + 8));
		// Offset to positive
		__m256i pos_nums1 = _mm256_add_epi32(v_nums1, OFFSET_V);
		__m256i pos_nums2 = _mm256_add_epi32(v_nums2, OFFSET_V);
		// Mask off key and shift to bottom of register
		__m256i ymmkeys1 = _mm256_srli_epi32(
			_mm256_and_epi32(pos_nums1, KEY_MASK_V),
			(int)key_shift);
		__m256i ymmkeys2 = _mm256_srli_epi32(
			_mm256_and_epi32(pos_nums2, KEY_MASK_V),
			(int)key_shift);

		// Due to store to load forwarding, we want these to be 32 bit
		// And moved into general purpose registers all at once
		alignas(32) uint32_t gpr_keys[16];

		_mm256_store_si256((__m256i*)&gpr_keys[0], ymmkeys1);
		_mm256_store_si256((__m256i*)&gpr_keys[8], ymmkeys2);

		// no inter-lane dependency, look ahead can optimize
		bkt_accumulators[gpr_keys[0] + (0 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[1] + (1 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[2] + (2 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[3] + (3 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[4] + (4 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[5] + (5 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[6] + (6 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[7] + (7 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[8] + (0 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[9] + (1 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[10] + (2 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[11] + (3 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[12] + (4 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[13] + (5 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[14] + (6 * NUM_BUCKETS)]++;
		bkt_accumulators[gpr_keys[15] + (7 * NUM_BUCKETS)]++;
	} // End for
	// TODO: tail

	uint16_t bkt_counts[NUM_BUCKETS] = {0};

	// Gather buckets

	// Store to hash

	// Compare highest magnitudes down

	retvals[0] = 0;
	retvals[1] = 1;

	*returnsize = 2;
	return retvals;
}

int main()
{
	int numsize = 4;
	int nums[] = {2, 7, 11, 15, 99, 3, 8, 9};
	// NOTE: Demonstrate the shifting key_mask
	int target = -1000000000;
	int retsize = 0;
	int* retvals = NULL;

	TIME(retvals = twoSum(nums, numsize, target, &retsize););

	if (retvals && nums[retvals[0]] + nums[retvals[1]] == target) {
		free(retvals);
		return 0;
	}
	free(retvals);
	return 1;
}

//==============================================================================
// Some consideration for repetitive code, tallying bucket counts
//------------------------------------------------------------------------------
// 	// X macro for repititous code
// 	// clang-format off
// #define BUCKET_LIST(X) \
// 	X(1)  X(2)  X(3)  X(4)  X(5)  X(6)  X(7)  X(8)  X(9)  X(10)
// X(11)      \
// 	X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19) X(20) X(21)
// X(22)      \ 	X(23) X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)
// 	// clang-format on
//
// 	// Make an accumulator ID for all 31 buckets
// #define DECLARE_CONST(i) const __m256i ID_##i = _mm256_set1_epi32(i);
// 	BUCKET_LIST(DECLARE_CONST)
//
// 	// Make an accumulator for all 31 buckets
// #define DECLARE_ACC(i) __m256i acc_##i = _mm256_setzero_si256();
// 	BUCKET_LIST(DECLARE_ACC)
//
// 	// Map data to positive range && filter data by magnitude
// 	for (int i = 0; i <= n - 8; i += 8) {
// 		__m256i v = _mm256_load_si256((__m256i*)&nums[i]);
// 		// Add offset to vector to map to positive range
// 		__m256i vp = _mm256_add_epi32(v, OFFSET_V);
// 		// Count leading zeros
// 		__m256i clz = clz_mm256_epi32(vp);
//
// #define UPDATE_ACC(i) \ 	acc_##i = _mm256_sub_epi32(acc_##i,
// _mm256_cmpeq_epi32(clz, ID_##i)); 		BUCKET_LIST(UPDATE_ACC)
// 	}

//==============================================================================
// Unused vectorize count leading 0's
//------------------------------------------------------------------------------
// /**
//  * @brief Count leading 0's by vector
//  *
//  * @param[in] f input vector. All vals must be > 0
//  * @return vector of leading zero counts
//  */
// __attribute__((target("avx2"))) //
// inline __m256i clz_mm256_ps(__m256 f)
// {
// 	const __m256i CLZ_MAGIC = _mm256_set1_epi32(158);
// 	const __m256i EXP_MASK = _mm256_set1_epi32(0xFF);
//
// 	// Isolate biased exponent from floats
// 	__m256i exp = _mm256_and_si256(
// 		// Shift right 23 bits & mask off low byte (exp bits)
// 		_mm256_srli_epi32(_mm256_castps_si256(f), 23),
// EXP_MASK);
//
// 	// Remove bias
// 	return _mm256_sub_epi32(CLZ_MAGIC, exp);
// }
//
// /**
//  * @brief Count leading 0's by vector (ints)
//  *
//  * @param[in] v input vector. All vals > 0
//  * @return vector of leading 0 counts
//  */
// __attribute__((target("avx2"))) //
// inline __m256i clz_mm256_epi32(__m256i v)
// {
// 	// Clear bottom byte to avoid rounding errors on large nums
// 	__m256i shift = _mm256_srli_epi32(v, 8);
// 	__m256i masked = _mm256_andnot_si256(shift, v);
// 	__m256 f = _mm256_cvtepi32_ps(masked);
// 	return clz_mm256_ps(f);
// }
