use isr_ion::geomagnetic::igrf14;
#[test]
fn matches_official_igrf14_reference() {
    let mut worst = 0f64;
    let mut count = 0;
    for line in include_str!("data/igrf-reference.txt").lines() {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let actual = igrf14(v[0], v[1], v[2], v[3]).unwrap();
        for (a, b) in actual.iter().zip(&v[4..]) {
            worst = worst.max((a - b).abs());
            assert!((a - b).abs() < 0.2, "{line}: {actual:?}");
        }
        count += 1;
    }
    assert_eq!(count, 240);
    eprintln!("{count} reference rows; worst field difference {worst} nT");
}
