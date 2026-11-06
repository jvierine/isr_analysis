//! Independent single-precision IRI Fortran reference, run on Compuntu.
use isr_ion::{composition::dy1985, topside::calion};
#[test]
fn matches_iri_reference_across_seasons_fluxes_and_latitudes() {
    let mut count = 0;
    let mut worst = 0f64;
    for line in include_str!("data/composition-reference.txt").lines() {
        let mut words = line.split_whitespace();
        let kind = words.next().unwrap();
        let v: Vec<f64> = words.map(|s| s.parse().unwrap()).collect();
        let (actual, expected) = if kind == "D" {
            (dy1985(v[0], 30., 18.34, 150., 11.5).unwrap(), &v[1..])
        } else {
            (
                calion(v[0], v[1], v[2], v[3] as u32, v[4]).unwrap(),
                &v[5..],
            )
        };
        for (a, b) in actual.iter().zip(expected) {
            let error = (a - b).abs();
            worst = worst.max(error);
            assert!(error < 2e-5, "{line}: Rust {actual:?}; error {error}");
        }
        count += 1;
    }
    assert_eq!(count, 1204);
    eprintln!("{count} IRI reference rows; worst fraction difference {worst}");
}
