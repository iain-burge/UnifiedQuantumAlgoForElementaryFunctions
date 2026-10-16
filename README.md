# A Unified Quantum Algorithm For Elementary Functions
A Unified and Universal CORDIC Based Algorithm to Efficiently Compute All Elementary Functions with Minimal Overhead.
Based on the classical extended version of Walther [1], a few tricks from Mazenc et al. [2], and the sneaky tricks that make it all reversible from Burge et al. [3].

### Abstract Sketch
Presented here is a universal CORDIC based algorithm that takes roughly 5n+O(1) space, O(n) additions or O(n^2) CNOT gates to perform the algorithm on n-bits.
Depending on two parameters, the algorithm can natively compute: x·y, x/y, sin x, cos x, arctan x, sinh x, cosh x, arctanh x, arcsin x, arcsinh x, x (identity).
Leveraging an addition or two extra, we can also perform ln x, e^x, and sqrt x.
Through multiple applications of CORDIC we can also compute: tan x, htan x, x^y.

### Explanation
The Universal Quantum CORDIC Draft pdf file gives a sketch of the related work and of the algorithm.

### Requirements:
- Python            3.14.7
- matplotlib        3.11.2
- numpy             2.5.3


### Note: 
Converting from the classical reversible algorithm to a quantum circuit is not finished, but will proceed identically to the methods used in the following REPO: https://github.com/iain-burge/QuantumCORDIC

### References
[1] Walther, John Stephen. "A unified algorithm for elementary functions." Proceedings of the May 18-20, 1971, spring joint computer conference. 1971.

[2] Mazenc, Christophe, Xavier Merrheim, and J-M. Muller. "Computing Functions cos/sup-1/and sin/sup-1/Using CORDIC." IEEE Transactions on Computers 42.1 (1993): 118-122.

[3] I. Burge, M. Barbeau, and J. Garcia-Alfaro, “Quantum Cordic – Arcsine on a Budget,” in 2026 IEEE International Conference on Quantum Computing and Engineering (QCE), vol. 1, pp. 1–6, IEEE, 2026.
