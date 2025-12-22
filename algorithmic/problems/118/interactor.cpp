#include "testlib.h"
#include <bits/stdc++.h>

using i64 = long long;

int main(int argc, char ** argv){
	registerInteraction(argc, argv);

	int id = inf.readInt(), t = inf.readInt();
	println(id, t);

	double ratio = 1, unbounded_ratio = INFINITY;
	while (t--) {
		i64 p = inf.readLong();
		int n = inf.readInt();
		println(p, n);

		i64 seed = inf.readLong(), y = inf.readLong();
		for (int q = 0; ; q++) {
			if (q > 200) {
				quitp(0., "Too many queries. Ratio: 0.0000");
			}

			std::string op = ouf.readToken();
			if (op == "?") {
				i64 x = ouf.readLong();
				if (x < 0 || x > i64(1e18)) {
					quitf(_wa, "Invalid query: element %d is out of range [0, 1000000000000000000]", x);
				}

				seed = seed * n % p;
				println((y > x ? 2 : y == x ? 1 : 0) ^ (seed % n));
			} else if (op == "!") {
				if (ouf.readLong() == y) {
					if (q > 100) {
						ratio = std::min(ratio, 1 - .7 * (q - 100) / 100);
					}
					unbounded_ratio = std::min(unbounded_ratio, 1 - .7 * (q - 100) / 100);
					break;
				} else {
					quitp(0., "Wrong guess. Ratio: 0.0000");
				}
			} else {
				quitf(_wa, "Invalid action type: expected ? or !, but got %s", op.c_str());
			}
		}
	}
	quitp(ratio, "Correct guess. Ratio: %.4f, RatioUnbounded: %.4f", ratio, unbounded_ratio);

	return 0;
}