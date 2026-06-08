use std::cmp::Ordering;

/// Tropical semiring element using max-plus arithmetic.
///
/// In the tropical semiring (ℝ ∪ {-∞}, max, +):
/// - "Addition" is max(a, b)
/// - "Multiplication" is a + b
/// - Additive identity (tropical zero) is -∞
/// - Multiplicative identity (tropical one) is 0.0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tropical(pub f64);

impl Eq for Tropical {}

impl PartialOrd for Tropical {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

/// Tropical "addition": max(a, b)
impl std::ops::Add for Tropical {
    type Output = Tropical;
    fn add(self, rhs: Self) -> Self::Output {
        Tropical(self.0.max(rhs.0))
    }
}

/// Tropical "multiplication": a + b
impl std::ops::Mul for Tropical {
    type Output = Tropical;
    fn mul(self, rhs: Self) -> Self::Output {
        Tropical(self.0 + rhs.0)
    }
}

impl Tropical {
    /// Tropical zero: the additive identity (-∞)
    pub fn tropical_zero() -> Self {
        Tropical(f64::NEG_INFINITY)
    }

    /// Tropical one: the multiplicative identity (0.0)
    pub fn tropical_one() -> Self {
        Tropical(0.0)
    }

    /// Convert from standard (real) to tropical: log(x)
    /// This is the log-semiring isomorphism.
    pub fn from_standard(x: f64) -> Self {
        assert!(x > 0.0, "from_standard requires positive input, got {x}");
        Tropical(x.ln())
    }

    /// Convert from tropical back to standard: exp(x)
    pub fn to_standard(self) -> f64 {
        self.0.exp()
    }

    /// Verify the idempotent property: max(a, a) = a
    /// This is THE defining property that makes tropical arithmetic "sparse".
    pub fn is_idempotent(self) -> bool {
        let sum = self + self;
        sum == self
    }

    /// Compute the tropical power: tropical exponentiation by repeated multiplication.
    /// Tropical a^n = n * a
    pub fn tpow(self, n: usize) -> Self {
        Tropical(self.0 * n as f64)
    }

    /// Tropical division (inverse of multiplication): subtraction.
    pub fn tdiv(self, rhs: Self) -> Self {
        Tropical(self.0 - rhs.0)
    }

    /// Inner value
    pub fn inner(self) -> f64 {
        self.0
    }

    /// Check if this is the tropical zero element
    pub fn is_tropical_zero(self) -> bool {
        self.0 == f64::NEG_INFINITY
    }

    /// Check if this is the tropical one element
    pub fn is_tropical_one(self) -> bool {
        self.0 == 0.0
    }
}

/// Tropical matrix multiplication: (A ⊗ B)[i][j] = max_k(A[i][k] + B[k][j])
///
/// This is the tropical analogue of standard matrix multiplication.
/// Complexity: O(n³) same as standard matmul, but with max/add instead of multiply/add.
pub fn tropical_matrix_mul(
    a: &[Vec<Tropical>],
    b: &[Vec<Tropical>],
) -> Vec<Vec<Tropical>> {
    let n = a.len();
    assert!(n > 0, "empty matrix a");
    let m = a[0].len();
    let p = b[0].len();
    assert_eq!(b.len(), m, "dimension mismatch: a cols ({m}) != b rows ({})", b.len());

    let mut result = vec![vec![Tropical::tropical_zero(); p]; n];

    for i in 0..n {
        for j in 0..p {
            let mut best = Tropical::tropical_zero();
            for k in 0..m {
                let product = a[i][k] * b[k][j]; // tropical mul = addition
                best = best + product;            // tropical add = max
            }
            result[i][j] = best;
        }
    }

    result
}

/// Compute the tropical rank of a matrix.
///
/// The tropical rank is defined as the size of the largest square submatrix
/// with a non-singular tropical determinant (i.e., the tropical determinant is
/// not tropical zero, meaning the optimal assignment is unique).
///
/// We use the definition based on tropical linear independence: a matrix has
/// tropical rank r if the largest r×r minor has a unique maximizing permutation.
pub fn tropical_rank(matrix: &[Vec<Tropical>]) -> usize {
    let n = matrix.len();
    if n == 0 {
        return 0;
    }
    let m = matrix[0].len();
    let min_dim = n.min(m);

    // Check rank from largest possible down
    for r in (1..=min_dim).rev() {
        if has_unique_maximizing_minor(matrix, r) {
            return r;
        }
    }
    0
}

/// Check if there exists an r×r minor with a unique maximizing permutation.
fn has_unique_maximizing_minor(matrix: &[Vec<Tropical>], r: usize) -> bool {
    let n = matrix.len();
    let m = matrix[0].len();

    // Generate all combinations of r rows and r columns
    let row_combos = combinations(n, r);
    let col_combos = combinations(m, r);

    for rows in &row_combos {
        for cols in &col_combos {
            if has_unique_max_perm(matrix, rows, cols) {
                return true;
            }
        }
    }
    false
}

/// Check if the submatrix defined by rows and cols has a unique maximizing permutation.
fn has_unique_max_perm(
    matrix: &[Vec<Tropical>],
    rows: &[usize],
    cols: &[usize],
) -> bool {
    let r = rows.len();
    let perms = permutations(r);

    let mut best_val = f64::NEG_INFINITY;
    let mut best_count = 0;

    for perm in &perms {
        let mut val = 0.0f64;
        for i in 0..r {
            val += matrix[rows[i]][cols[perm[i]]].0;
        }
        match val.partial_cmp(&best_val) {
            Some(Ordering::Greater) => {
                best_val = val;
                best_count = 1;
            }
            Some(Ordering::Equal) => {
                best_count += 1;
            }
            _ => {}
        }
    }

    best_count == 1 && best_val > f64::NEG_INFINITY
}

/// Generate all k-combinations of n items
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();
    generate_combos(n, k, 0, &mut current, &mut result);
    result
}

fn generate_combos(
    n: usize, k: usize, start: usize,
    current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    for i in start..n {
        current.push(i);
        generate_combos(n, k, i + 1, current, result);
        current.pop();
    }
}

/// Generate all permutations of n items
fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut items: Vec<usize> = (0..n).collect();
    let mut result = Vec::new();
    permute(&mut items, 0, &mut result);
    result
}

fn permute(items: &mut Vec<usize>, start: usize, result: &mut Vec<Vec<usize>>) {
    if start == items.len() {
        result.push(items.clone());
        return;
    }
    for i in start..items.len() {
        items.swap(start, i);
        permute(items, start + 1, result);
        items.swap(start, i);
    }
}

/// Tropical determinant: max over all permutations π of Σ a[i][π(i)]
pub fn tropical_determinant(matrix: &[Vec<Tropical>]) -> Tropical {
    let n = matrix.len();
    assert!(n > 0, "empty matrix");
    for row in matrix {
        assert_eq!(row.len(), n, "determinant requires square matrix");
    }

    let perms = permutations(n);
    let mut best = Tropical::tropical_zero();

    for perm in &perms {
        let mut val = Tropical::tropical_one();
        for i in 0..n {
            val = val * matrix[i][perm[i]];
        }
        best = best + val;
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_add_is_max() {
        let a = Tropical(3.0);
        let b = Tropical(7.0);
        assert_eq!(a + b, Tropical(7.0));
        assert_eq!(b + a, Tropical(7.0));
    }

    #[test]
    fn test_tropical_mul_is_add() {
        let a = Tropical(3.0);
        let b = Tropical(7.0);
        assert_eq!(a * b, Tropical(10.0));
    }

    #[test]
    fn test_tropical_zero_identity() {
        let z = Tropical::tropical_zero();
        let a = Tropical(5.0);
        // a + z = max(a, -∞) = a
        assert_eq!(a + z, a);
        assert_eq!(z + a, a);
    }

    #[test]
    fn test_tropical_one_identity() {
        let one = Tropical::tropical_one();
        let a = Tropical(5.0);
        // a * one = a + 0 = a
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);
    }

    #[test]
    fn test_idempotent_property() {
        let a = Tropical(3.14);
        assert!(a.is_idempotent());
        let b = Tropical(-100.0);
        assert!(b.is_idempotent());
        let z = Tropical::tropical_zero();
        assert!(z.is_idempotent());
    }

    #[test]
    fn test_from_to_standard_roundtrip() {
        let x = 2.5f64;
        let tropical = Tropical::from_standard(x);
        let recovered = tropical.to_standard();
        assert!((recovered - x).abs() < 1e-10);
    }

    #[test]
    fn test_from_standard_log() {
        // from_standard(x) = log(x)
        let t = Tropical::from_standard(std::f64::consts::E);
        assert!((t.0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_standard_exp() {
        // to_standard(log(x)) = x
        let t = Tropical(0.0);
        assert!((t.to_standard() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tropical_matrix_mul_identity() {
        // I ⊗ A = A where I is tropical identity (0 on diagonal, -∞ elsewhere)
        let n = 3;
        let mut identity = vec![vec![Tropical::tropical_zero(); n]; n];
        for i in 0..n {
            identity[i][i] = Tropical::tropical_one();
        }

        let a = vec![
            vec![Tropical(1.0), Tropical(2.0), Tropical(3.0)],
            vec![Tropical(4.0), Tropical(5.0), Tropical(6.0)],
            vec![Tropical(7.0), Tropical(8.0), Tropical(9.0)],
        ];

        let result = tropical_matrix_mul(&identity, &a);
        for i in 0..n {
            for j in 0..n {
                assert!((result[i][j].0 - a[i][j].0).abs() < 1e-10,
                    "mismatch at ({i},{j}): got {}, expected {}", result[i][j].0, a[i][j].0);
            }
        }
    }

    #[test]
    fn test_tropical_matrix_mul_known() {
        let a = vec![
            vec![Tropical(1.0), Tropical(2.0)],
            vec![Tropical(3.0), Tropical(4.0)],
        ];
        let b = vec![
            vec![Tropical(5.0), Tropical(6.0)],
            vec![Tropical(7.0), Tropical(8.0)],
        ];

        // (A ⊗ B)[0][0] = max(1+5, 2+7) = max(6, 9) = 9
        // (A ⊗ B)[0][1] = max(1+6, 2+8) = max(7, 10) = 10
        // (A ⊗ B)[1][0] = max(3+5, 4+7) = max(8, 11) = 11
        // (A ⊗ B)[1][1] = max(3+6, 4+8) = max(9, 12) = 12
        let result = tropical_matrix_mul(&a, &b);
        assert_eq!(result[0][0], Tropical(9.0));
        assert_eq!(result[0][1], Tropical(10.0));
        assert_eq!(result[1][0], Tropical(11.0));
        assert_eq!(result[1][1], Tropical(12.0));
    }

    #[test]
    fn test_tropical_rank_identity() {
        let mut identity = vec![vec![Tropical::tropical_zero(); 3]; 3];
        for i in 0..3 {
            identity[i][i] = Tropical::tropical_one();
        }
        assert_eq!(tropical_rank(&identity), 3);
    }

    #[test]
    fn test_tropical_rank_rank1() {
        // All rows identical → rank 1
        let matrix = vec![
            vec![Tropical(1.0), Tropical(2.0)],
            vec![Tropical(1.0), Tropical(2.0)],
        ];
        assert_eq!(tropical_rank(&matrix), 1);
    }

    #[test]
    fn test_tropical_rank_full() {
        let matrix = vec![
            vec![Tropical(1.0), Tropical(0.0)],
            vec![Tropical(0.0), Tropical(1.0)],
        ];
        assert_eq!(tropical_rank(&matrix), 2);
    }

    #[test]
    fn test_tropical_determinant() {
        let matrix = vec![
            vec![Tropical(1.0), Tropical(2.0)],
            vec![Tropical(3.0), Tropical(4.0)],
        ];
        // det = max(1+4, 2+3) = max(5, 5) = 5
        let det = tropical_determinant(&matrix);
        assert!((det.0 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_tropical_pow() {
        let a = Tropical(3.0);
        assert_eq!(a.tpow(0), Tropical::tropical_one());
        assert_eq!(a.tpow(1), Tropical(3.0));
        assert_eq!(a.tpow(2), Tropical(6.0));
        assert_eq!(a.tpow(3), Tropical(9.0));
    }

    #[test]
    fn test_tropical_div() {
        let a = Tropical(10.0);
        let b = Tropical(3.0);
        assert_eq!(a.tdiv(b), Tropical(7.0));
    }

    #[test]
    fn test_tropical_zero_is_zero() {
        assert!(Tropical::tropical_zero().is_tropical_zero());
        assert!(!Tropical(0.0).is_tropical_zero());
    }

    #[test]
    fn test_tropical_one_is_one() {
        assert!(Tropical::tropical_one().is_tropical_one());
        assert!(!Tropical(5.0).is_tropical_one());
    }
}
