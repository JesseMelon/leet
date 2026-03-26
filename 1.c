#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdalign.h>
#include <immintrin.h>

// TODO: store packed vector into keys

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
#define VLOG(vec)                                                              \
	do {                                                                   \
		alignas(32) int c[8];                                          \
		_mm256_store_si256((__m256i*)c, vec);                          \
		printf("%s: %d %d %d %d %d %d %d %d\n", #vec, c[0], c[1],      \
		       c[2], c[3], c[4], c[5], c[6], c[7]);                    \
	} while (0)
#define VLOGB(vec)                                                             \
	do {                                                                   \
		alignas(32) int c[8];                                          \
		_mm256_store_si256((__m256i*)c, vec);                          \
		printf("%s:\n%10u\t%032b\n%10u\t%032b\n%10u\t%032b\n%10u\t%"   \
		       "032b\n%10u\t"                                          \
		       "%032b\n%10u\t%032b\n%10u\t%032b\n%10u\t%032b\n",       \
		       #vec, c[0], c[0], c[1], c[1], c[2], c[2], c[3], c[3],   \
		       c[4], c[4], c[5], c[5], c[6], c[6], c[7], c[7]);        \
	} while (0)
#else
#define SYS_EVAL()
#define TIME(code) code
#define CHECK_ALIGN(ptr, bytes)
#define LOG(msg)
#define VLOG(vec)
#define VLOGB(vec)
#endif

#define N_MAX 10000
#define OFFSET 1000000001
#define KEY_BITS 11U		 // TODO: Make dynamic?
#define ODD_BITS (KEY_BITS - 2U) // section of key from num
#define NUM_BUCKETS 2048U	 // 1 << KEY_BITS

inline void odd_prefix(int* nums, int n, int target, uint32_t ptgt,
		       uint16_t tg_key, uint8_t key_shift, uint32_t num_mask);
inline void even_prefix(int* nums, int n, int target, uint32_t ptgt,
			uint16_t tg_key, uint8_t key_shift, uint32_t num_mask);

/*==============================================================================
 *
 * TWO SUM
 *
 * - applies constant offset to entire problem set
 * --- allows existence culling based on magnitude
 *
 * - custom hash
 * --- 11 bits, 1 carry indicator, then vals correlated to 10 msb bits of target
------------------------------------------------------------------------------*/
__attribute__((target("avx2"))) //
int* twoSum(int* nums, int n, int target, int* returnsize)
{
	// SYS_EVAL();			// 8MB stack, avx2 supported
	// CHECK_ALIGN(nums, 32);	// nums is aligned if large

	// Heap alloc return values
	int* retvals = malloc(sizeof(int) * 2);

	// Return on invalid inputs
	if (retvals == NULL || n > N_MAX) {
		*returnsize = 0;
		return NULL;
	}

	// Shift target such that all values are positive
	uint32_t ptgt = target + (2 * OFFSET);
	LOG("Target %u is now shifted to %u, %032b\n", target, ptgt, ptgt);

	// calculate the shift needed to capture top 9 msb of target
	uint8_t key_shift;
	{

		// Count leading zeroes
		uint8_t lz = __builtin_clz(ptgt);
		LOG("Target has %u leading zeros\n", lz);

		// Determine position of msb
		uint8_t msb_pos = 32 - lz;
		LOG("Position of MSB is %u\n", msb_pos - 1);

		// Shift = position of most significant bits of target or bottom
		key_shift = (msb_pos > KEY_BITS) ? msb_pos - ODD_BITS : 0;
		LOG("Key will be shifted by %u\n", key_shift);
		LOG("1U << keyshift: %032b\n", 1U << key_shift);
	}

	// All keybits but two will be a mask of nums
	uint32_t num_mask = ((1U << ODD_BITS) - 1U) << (key_shift);
	LOG("Key mask is:\t%032b\n", num_mask);

	// Calculate target key
	uint16_t tg_key = ((ptgt & num_mask) >> key_shift) |
			  ((uint16_t)target & 1U) << ODD_BITS;
	LOG("Target key: %u, %016b\n", tg_key, tg_key);

	// if ((uint)target & 1U << (key_shift - 1U)) {
	// 	LOG("Prefix is odd\n");
	// 	odd_prefix(nums, n, target, ptgt, tg_key, key_shift, num_mask);
	// } else {
	// 	LOG("Prefix is even\n");
	// 	even_prefix(nums, n, target, ptgt, tg_key, key_shift, num_mask);
	// }

	// Most signif keybit is complement carry in flag
	uint32_t carry_bit = 1U << (key_shift - 1U);
	LOG("Low bit is:\t%032b\n", carry_bit);

	const __m256i OFFSET_V = _mm256_set1_epi32(OFFSET);
	const __m256i KEY_MASK_V = _mm256_set1_epi32((int)num_mask);
	const __m256i LOW_BIT_V = _mm256_set1_epi32((int)carry_bit);
	const __m256i ONE_BIT_V = _mm256_set1_epi32(1U);
	const __m256i HIGH_BIT_V = _mm256_set1_epi32(1U << (KEY_BITS - 1U));
	const __m256i EVEN_BIT_V = _mm256_set1_epi32(1U << (KEY_BITS - 2U));
	const __m256i TARGET_V = _mm256_set1_epi32((int)ptgt);

	// bkt_idx & bkt_counts share memory
	uint16_t bkt_counts[NUM_BUCKETS] = {0}; // ~4kb
	uint16_t keys[N_MAX];			// 16kb

	// Start a new scope block because accs uses 32+ KB ( With 11+ bit
	// key), and doesnt need to persist for the entire function. This
	// reduces the total stack depth for if we need other arrays later, and
	// helps to stay in cache somewhat. TODO: Measure this?
	{
		// 8 long arrays
		uint16_t accs[8][NUM_BUCKETS] = {0};

		for (int i = 0; i < n; i += 16) {

			// Load 8 nums
			__m256i vnums1 =
				_mm256_loadu_si256((__m256i*)(nums + i));
			__m256i vnums2 =
				_mm256_loadu_si256((__m256i*)(nums + i + 8));

			// Offset to positive
			__m256i pnums1 = _mm256_add_epi32(vnums1, OFFSET_V);
			__m256i pnums2 = _mm256_add_epi32(vnums2, OFFSET_V);

			// Mask off key and shift to bottom of register
			__m256i mask_nums1 =
				_mm256_and_si256(pnums1, KEY_MASK_V);
			__m256i mask_nums2 =
				_mm256_and_si256(pnums2, KEY_MASK_V);

			// Calculate complement of nums[i]
			__m256i compls1 = _mm256_sub_epi32(TARGET_V, pnums1);
			__m256i compls2 = _mm256_sub_epi32(TARGET_V, pnums2);
			VLOGB(compls1);

			// Determine whether a carry is added to prefix
			__m256i shared_bits1 =
				_mm256_and_si256(compls1, pnums1);
			VLOGB(shared_bits1);
			__m256i shared_bits2 =
				_mm256_and_si256(compls2, pnums2);
			__m256i low_bit1 =
				_mm256_and_si256(shared_bits1, LOW_BIT_V);
			VLOGB(low_bit1);
			__m256i low_bit2 =
				_mm256_and_si256(shared_bits2, LOW_BIT_V);
			__m256i carry1 =
				_mm256_cmpeq_epi32(low_bit1, LOW_BIT_V);
			VLOGB(carry1);
			__m256i carry2 =
				_mm256_cmpeq_epi32(low_bit2, LOW_BIT_V);

			// Is the num even or odd
			__m256i odd1 = _mm256_cmpeq_epi32(
				_mm256_and_si256(vnums1, ONE_BIT_V), ONE_BIT_V);
			VLOGB(odd1);
			__m256i odd2 = _mm256_cmpeq_epi32(
				_mm256_and_si256(vnums2, ONE_BIT_V), ONE_BIT_V);

			// Shift 10 bit keys to bottom
			__m256i shift_nums1 =
				_mm256_srli_epi32(mask_nums1, key_shift);
			__m256i shift_nums2 =
				_mm256_srli_epi32(mask_nums2, key_shift);

			// Apply even/odd bit to key
			__m256i even_key1 = _mm256_or_si256(
				_mm256_and_si256(odd1, EVEN_BIT_V),
				shift_nums1);
			__m256i even_key2 = _mm256_or_si256(
				_mm256_and_si256(odd2, EVEN_BIT_V),
				shift_nums2);

			// Apply carry bit to key
			__m256i ymmkeys1 = _mm256_or_si256(
				_mm256_and_si256(carry1, HIGH_BIT_V),
				even_key1);
			__m256i ymmkeys2 = _mm256_or_si256(
				_mm256_and_si256(carry2, HIGH_BIT_V),
				even_key2);

			// Keys are now ready

			// pack keys in keys[]
			// Permute is needed because SIMD uses 128 bit lanes
			__m256i packed = _mm256_permute4x64_epi64(
				// Pack packs the two 32 bit vectors lanewise
				_mm256_packus_epi32(ymmkeys1, ymmkeys2),
				// 2 bit destination blocks L to R
				// 3 -> 3
				// 2 -> 1
				// 1 -> 2
				// 0 -> 0
				// this reverses the lanewiseness of the perm
				0b11011000);

			// TODO: store packed into keys

			// Due to store to load forwarding, we want
			// these to be 32 bit And moved into general
			// purpose registers once
			alignas(32) uint32_t gpr_keys[16];

			_mm256_store_si256((__m256i*)&gpr_keys[0], ymmkeys1);
			_mm256_store_si256((__m256i*)&gpr_keys[8], ymmkeys2);

			for (int j = 0; j < 16; j++) {
				LOG("entry %u shifted: %u, %032b. Key: %u, "
				    "%032b\n",
				    nums[i + j], nums[i + j] + OFFSET,
				    nums[i + j] + OFFSET, gpr_keys[j],
				    gpr_keys[j]);
			}

			// no inter-lane dependency, look ahead can optimize
			// NOTE: If data were larger, tiling would be better
			// Tiling would use 8 * 16 buckets sets, and use mod 8
			// indices via '& 7' paired with '>> 3' during indexing
			accs[0][gpr_keys[0]]++;
			accs[1][gpr_keys[1]]++;
			accs[2][gpr_keys[2]]++;
			accs[3][gpr_keys[3]]++;
			accs[4][gpr_keys[4]]++;
			accs[5][gpr_keys[5]]++;
			accs[6][gpr_keys[6]]++;
			accs[7][gpr_keys[7]]++;
			accs[0][gpr_keys[8]]++;
			accs[1][gpr_keys[9]]++;
			accs[2][gpr_keys[10]]++;
			accs[3][gpr_keys[11]]++;
			accs[4][gpr_keys[12]]++;
			accs[5][gpr_keys[13]]++;
			accs[6][gpr_keys[14]]++;
			accs[7][gpr_keys[15]]++;

		} // End for
		// TODO: tail

		// reduce final bucket counts
		for (uint32_t b = 0; b < NUM_BUCKETS; b += 16) {
			__m256i t = _mm256_setzero_si256();

			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[0][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[1][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[2][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[3][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[4][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[5][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[6][b]));
			t = _mm256_add_epi16(
				t, _mm256_loadu_si256((__m256i*)&accs[7][b]));
			_mm256_storeu_si256((__m256i*)(bkt_counts + b), t);
		}
	} // end scope block

	// Loop for logging only
	for (int a = 1260, b = a; b < a + 10; b++) {
		LOG("bucket %u = %u\n", b, bkt_counts[b]);
	}

	// Store to hash (stack, obvi)
	// uint8_t fingerp[n];
	// uint32_t payload[n];

	// Used to quickly check a bucket is considered before value to hash
	uint64_t bkt_exist_bits[NUM_BUCKETS / 64] = {0};

	uint16_t offset = 0, count;

	// Need for bkt idx, and bkt counts are m-xclusive, so save ~4KB stack
	uint16_t* bkt_idx = bkt_counts;

	// Store complements adjacent
	// Decompose target key into two solution keys
	// Start at target key, and work down (ignoring keys above tgt)

	// tg_key / 2 has same complement
	uint16_t key = tg_key >> 1U;
	uint16_t rem_mod2 = tg_key & 1U;

	// TODO: decompose key based on even/odd
	if (rem_mod2) {
		// target decomposes into two key pairs, carry & ncarry
	} else {
		// target decomposes into just one key pair
	}

	count = bkt_counts[key];
	// set the exists bit
	bkt_exist_bits[key >> 6U] |= (uint64_t)(count != 0) << (key & 63U);
	bkt_idx[key] = offset;
	offset += count;

	retvals[0] = 0;
	retvals[1] = 1;

	*returnsize = 2;
	return retvals;
}

int main()
{
	int numsize = 16;
	int nums[] = {2,
		      7,
		      1025 - OFFSET,
		      1026 - OFFSET,
		      99 - OFFSET,
		      3 - (OFFSET / 2),
		      8 - (OFFSET / 4),
		      9 - (OFFSET / 8),
		      100,
		      101,
		      102,
		      103,
		      1,
		      4,
		      5,
		      6};
	// NOTE: Demonstrate the shifting key_mask
	int target = 2051 - (2 * OFFSET);
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
// X(22)      \ 	X(23) X(24) X(25) X(26) X(27) X(28) X(29) X(30)
// X(31)
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
