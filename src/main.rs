use rand::Rng;

pub struct Perceptron {
    pub(crate) weights: Vec<f64>,
    learning_rate: f64,
}

impl Perceptron {
    pub fn new(n_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        let weights = (0..=n_features)
            .map(|_| rng.random_range(-0.5..0.5))  // More concentrated initialization
            .collect();
        
        Self {
            weights,
            learning_rate,
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let sum = self.weights[0] + features.iter()
            .zip(&self.weights[1..])
            .map(|(&x, &w)| x * w)
            .sum::<f64>();

        (sum >= 0.0) as u8 as f64  // More idiomatic conversion
    }

    pub fn train(&mut self, inputs: &[&[f64]], targets: &[f64], epochs: usize) {
        for _ in 0..epochs {
            for (&input, &target) in inputs.iter().zip(targets) {
                let prediction = self.predict(input);
                let error = target - prediction;

                // Update bias
                self.weights[0] += self.learning_rate * error;

                // Update weights using zip for safety
                for (w, &x) in self.weights[1..].iter_mut().zip(input) {
                    *w += self.learning_rate * error * x;
                }
            }
        }
    }
}

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![0.0, 1.0, 1.0, 1.0];
    
    let input_slices: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();
    
    let mut perceptron = Perceptron::new(2, 0.1);
    perceptron.train(&input_slices, &targets, 100);

    println!("Trained weights: {:?}", perceptron.weights);
    
    for input in &inputs {
        let prediction = perceptron.predict(input);
        println!("{:?} -> {}", input, prediction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let p = Perceptron::new(3, 0.1);
        assert_eq!(p.weights.len(), 4);  // 3 features + 1 bias
        for w in p.weights {
            assert!(w >= -0.5 && w < 0.5, "Weight {} out of initialization range", w);
        }
    }

    #[test]
    fn test_predict() {
        let mut p = Perceptron::new(2, 0.1);
        // Manually set weights: bias = -0.3, w1 = 0.5, w2 = 0.5
        p.weights = vec![-0.3, 0.5, 0.5];
        
        assert_eq!(p.predict(&[0.0, 0.0]), 0.0);
        assert_eq!(p.predict(&[1.0, 0.0]), 1.0);
        assert_eq!(p.predict(&[0.0, 1.0]), 1.0);
        assert_eq!(p.predict(&[1.0, 1.0]), 1.0);
    }

    #[test]
    fn test_or_problem() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![0.0, 1.0, 1.0, 1.0];
        let input_slices: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();

        let mut p = Perceptron::new(2, 0.1);
        p.train(&input_slices, &targets, 100);

        for (input, &target) in inputs.iter().zip(&targets) {
            assert_eq!(p.predict(input), target, "Failed on input: {:?}", input);
        }
    }

    #[test]
    fn test_and_problem() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![0.0, 0.0, 0.0, 1.0];
        let input_slices: Vec<&[f64]> = inputs.iter().map(|v| v.as_slice()).collect();

        let mut p = Perceptron::new(2, 0.1);
        p.train(&input_slices, &targets, 1000);  // Needs more epochs to converge

        for (input, &target) in inputs.iter().zip(&targets) {
            assert_eq!(p.predict(input), target, "Failed on input: {:?}", input);
        }
    }
}