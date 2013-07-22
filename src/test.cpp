#include <iostream>
#include <random>
#include <vector>

#include "bandit.hpp"

using namespace std;

int main() {

	mt19937 rng;
	
	vector<bandit::dist_type> dists;
	for (int i = 0; i < 5; ++i) {
		dists.push_back(bandit::dist_type(i, 1));
	}

	bandit b(dists);

	for (int i = 0; i < 100; i++) {
		cout << b.pull_arm(rng, 0) << endl;
	}

}

